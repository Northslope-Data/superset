# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import io
import logging
import time
from typing import TYPE_CHECKING

from celery import current_task
from PIL import Image

logger = logging.getLogger(__name__)

# Time to wait after scrolling for content to settle and load (in milliseconds)
SCROLL_SETTLE_TIMEOUT_MS = 1000

# Fallback wall-clock budget, in seconds, for the entire tiled-screenshot
# operation (element lookup plus all per-tile spinner/animation waits
# combined), used when the budget can't be derived from the currently
# running Celery task's own time limit (see _resolve_wait_budget_seconds).
# Each tile's wait is capped at whatever remains of the budget so a slow
# dashboard degrades gracefully instead of running past the task's time
# limit and getting SIGKILLed mid-capture.
#
# A static config value isn't a reliable substitute for runtime derivation:
# superset/tasks/scheduler.py sets a report's limit per-schedule via
# `apply_async(time_limit=..., soft_time_limit=...)`, and per-task
# `task_annotations` (e.g. an operator giving thumbnail tasks a much
# shorter limit than reports) are Celery worker configuration invisible to
# any Superset config key -- only the running task itself knows its
# effective limit.
#
# Production has observed a Celery hard task_time_limit of 1740s (29 min)
# for report execution (2026-07-13 incident: a tiled screenshot was killed
# mid-capture with SoftTimeLimitExceeded). This fallback leaves a 300s
# margin under that ceiling for the rest of the pipeline that runs after
# tiling completes: combining tiles into one image, building the PDF, and
# uploading/delivering the notification. It applies when there's no Celery
# task context at all (e.g. synchronous thumbnail generation) or the task
# exposes no usable limit.
TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS = 1440  # 1740s limit - 300s margin

# Safety margin taken off a runtime-derived budget: min(this cap, this
# fraction of the task's own limit). The 300s cap matches the fallback
# budget's margin for large limits (e.g. the 1740s report limit); the
# fraction scales the margin down for small limits (e.g. a 120s thumbnail
# task limit) so it doesn't eat the whole budget or push it negative.
TILED_SCREENSHOT_BUDGET_MARGIN_FRACTION = 0.2
TILED_SCREENSHOT_BUDGET_MAX_MARGIN_SECONDS = 300

# Floor for a runtime-derived budget. Guards against a pathologically small
# task limit (well under a minute) yielding a near-zero or negative budget
# that would abort before capturing a single tile; a small positive budget
# is still capped by -- and will still be killed by -- the task's actual
# limit if it's smaller than this floor, but at least gives the tile loop a
# chance to capture what it can before that happens.
TILED_SCREENSHOT_MIN_BUDGET_SECONDS = 30


class TiledScreenshotBudgetExceededError(RuntimeError):
    """Raised when the tiled-screenshot time budget runs out mid-capture."""


def _resolve_wait_budget_seconds(log_context: str | None = None) -> float:
    """
    Derive the tiled-screenshot time budget from the currently running
    Celery task's own soft/hard time limit, if one is running and exposes
    one. This reflects per-task overrides (e.g. task_annotations giving
    thumbnail tasks a shorter limit than reports) that no static config
    value can see, since those overrides only exist at the Celery worker/
    runtime level.

    Falls back to TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS if there's no
    task context, the task exposes no usable limit, or anything goes wrong
    while inspecting it -- this must never be able to break a screenshot.
    """
    context_suffix = f" [{log_context}]" if log_context else ""
    try:
        if current_task:
            soft_limit, hard_limit = current_task.request.timelimit or (
                None,
                None,
            )
            limit = soft_limit or hard_limit
            if limit:
                margin = min(
                    TILED_SCREENSHOT_BUDGET_MAX_MARGIN_SECONDS,
                    limit * TILED_SCREENSHOT_BUDGET_MARGIN_FRACTION,
                )
                budget = max(TILED_SCREENSHOT_MIN_BUDGET_SECONDS, limit - margin)
                logger.info(
                    "Tiled screenshot budget derived from Celery task %s=%.1fs: "
                    "%.1fs (margin=%.1fs).%s",
                    "soft_time_limit" if soft_limit else "time_limit",
                    limit,
                    budget,
                    margin,
                    context_suffix,
                )
                return budget
    except Exception:
        logger.debug(
            "Failed to derive tiled screenshot budget from the Celery task "
            "context; using fallback budget of %ss.%s",
            TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS,
            context_suffix,
            exc_info=True,
        )
        return TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS

    logger.debug(
        "No usable Celery task time limit found; using fallback tiled "
        "screenshot budget of %ss.%s",
        TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS,
        context_suffix,
    )
    return TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS


try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
except ImportError:
    PlaywrightTimeout = Exception

if TYPE_CHECKING:
    try:
        from playwright.sync_api import Page
    except ImportError:
        Page = None


def combine_screenshot_tiles(
    screenshot_tiles: list[bytes], log_context: str | None = None
) -> bytes:
    """
    Combine multiple screenshot tiles into a single vertical image.

    Args:
        screenshot_tiles: List of screenshot bytes in PNG format
        log_context: Optional identifier (e.g. report execution id, or a
            thumbnail cache key) appended to log lines for tracing.

    Returns:
        Combined screenshot as bytes
    """
    if not screenshot_tiles:
        return b""

    if len(screenshot_tiles) == 1:
        return screenshot_tiles[0]

    context_suffix = f" [{log_context}]" if log_context else ""
    try:
        # Open all images
        images = [Image.open(io.BytesIO(tile)) for tile in screenshot_tiles]

        # Calculate total dimensions
        total_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        # Create combined image
        combined = Image.new("RGB", (total_width, total_height), "white")

        # Paste each tile
        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height

        # Convert back to bytes
        output = io.BytesIO()
        combined.save(output, format="PNG")
        return output.getvalue()

    except Exception as e:
        logger.exception("Failed to combine screenshot tiles: %s%s", e, context_suffix)
        # Return the first tile as fallback
        return screenshot_tiles[0]


def take_tiled_screenshot(
    page: "Page",
    element_name: str,
    tile_height: int,
    load_wait: int = 60,
    animation_wait: int = 0,
    log_context: str | None = None,
) -> bytes | None:
    """
    Take a tiled screenshot of a large dashboard by scrolling and capturing sections.

    Args:
        page: Playwright page object
        element_name: CSS class name of the element to screenshot
        tile_height: Height of each tile in pixels
        load_wait: Seconds to wait for charts to load per tile (default 60)
        animation_wait: Seconds to wait for chart animations per tile (default 0)
        log_context: Optional identifier (e.g. report execution id, or a
            thumbnail cache key) appended to log lines so a slow/timed-out
            capture can be traced back to the run that produced it.

    Returns:
        Combined screenshot bytes or None if failed

    Raises:
        TiledScreenshotBudgetExceededError: If the total time budget for the
            tiled-screenshot operation runs out before every tile has been
            verifiably captured. Callers must treat this as a hard failure
            rather than fall back to an unchecked/partial screenshot.
    """
    context_suffix = f" [{log_context}]" if log_context else ""
    wait_budget_seconds = _resolve_wait_budget_seconds(log_context)
    start_time = time.monotonic()
    try:
        # Get the target element
        element = page.locator(f".{element_name}")
        element.wait_for(timeout=30000)  # 30 second timeout

        # Get dashboard dimensions and position
        element_info = page.evaluate(f"""() => {{
            const el = document.querySelector(".{element_name}");
            const rect = el.getBoundingClientRect();
            return {{
                width: el.scrollWidth,
                height: el.scrollHeight,
                left: rect.left + window.scrollX,
                top: rect.top + window.scrollY,
            }};
        }}""")

        dashboard_width = element_info["width"]
        dashboard_height = element_info["height"]
        dashboard_left = element_info["left"]
        dashboard_top = element_info["top"]

        logger.info(
            "Dashboard: %sx%spx at (%s, %s)%s",
            dashboard_width,
            dashboard_height,
            dashboard_left,
            dashboard_top,
            context_suffix,
        )

        # Calculate number of tiles needed
        num_tiles = max(1, (dashboard_height + tile_height - 1) // tile_height)
        logger.info("Taking %s screenshot tiles%s", num_tiles, context_suffix)

        screenshot_tiles: list[bytes] = []

        for i in range(num_tiles):
            # Check the time budget before starting this tile's readiness wait.
            # If it's already exhausted, we can no longer verify this (or any
            # later) tile is actually ready to capture -- fail loudly instead
            # of silently snapshotting a spinner or blank chart, or running
            # past the Celery task time limit and getting SIGKILLed.
            tile_start = time.monotonic()
            elapsed = tile_start - start_time
            remaining_budget = wait_budget_seconds - elapsed
            if remaining_budget <= 0:
                # A customer-side chart-loading issue (a slow/hung dashboard),
                # not a Superset system fault, so this is a WARNING rather
                # than an ERROR -- consistent with #38130/#38441, which
                # deliberately downgraded screenshot timeout logs the same way.
                logger.warning(
                    "Tiled screenshot time budget exhausted on tile %s/%s: "
                    "%s/%s tiles captured so far, %.1fs elapsed of a %.1fs "
                    "budget. Aborting instead of capturing remaining tiles "
                    "unchecked.%s",
                    i + 1,
                    num_tiles,
                    len(screenshot_tiles),
                    num_tiles,
                    elapsed,
                    wait_budget_seconds,
                    context_suffix,
                )
                raise TiledScreenshotBudgetExceededError(
                    f"Tiled screenshot budget of "
                    f"{wait_budget_seconds:.1f}s exhausted "
                    f"after {len(screenshot_tiles)}/{num_tiles} tiles"
                )

            # Calculate scroll position to show this tile's content
            scroll_y = dashboard_top + (i * tile_height)

            page.evaluate(f"window.scrollTo(0, {scroll_y})")
            logger.debug(
                "Scrolled window to %s for tile %s/%s%s",
                scroll_y,
                i + 1,
                num_tiles,
                context_suffix,
            )
            # Wait for scroll to settle and content to load
            page.wait_for_timeout(SCROLL_SETTLE_TIMEOUT_MS)
            # Wait for any loading spinners visible in the current viewport to clear,
            # capped at whatever remains of the total time budget so a slow
            # dashboard degrades gracefully instead of exceeding it.
            # Only check viewport-visible spinners to avoid blocking on
            # virtualization placeholders rendered for off-screen charts.
            tile_load_wait = min(load_wait, remaining_budget)
            try:
                page.wait_for_function(
                    """() => {
                        const els = document.querySelectorAll('.loading');
                        for (const el of els) {
                            const r = el.getBoundingClientRect();
                            if (r.top < window.innerHeight && r.bottom > 0) {
                                return false;
                            }
                        }
                        return true;
                    }""",
                    timeout=tile_load_wait * 1000,
                )
            except PlaywrightTimeout:
                # Customer chart-loading timeout, not a system fault -- WARNING,
                # matching #38130/#38441 (see the budget-exhaustion log above).
                logger.warning(
                    "Timed out waiting for visible spinners to clear on tile "
                    "%s/%s (waited %.1fs of a %ss requested load_wait; %.1fs "
                    "elapsed of a %.1fs total budget; %s/%s tiles captured so "
                    "far).%s",
                    i + 1,
                    num_tiles,
                    tile_load_wait,
                    load_wait,
                    elapsed,
                    wait_budget_seconds,
                    len(screenshot_tiles),
                    num_tiles,
                    context_suffix,
                )
            spinner_wait_elapsed = time.monotonic() - tile_start

            # Wait for chart animations (e.g. ECharts) to finish after spinner clears.
            # The global animation wait before tiling only covers the first tile;
            # subsequent tiles need their own wait after data loads. Capped at
            # whatever remains of the budget; unlike the spinner wait above this
            # is cosmetic settling, not a readiness check, so we simply skip it
            # (rather than raise) once the budget runs out.
            animation_wait_elapsed = 0.0
            if animation_wait > 0:
                elapsed = time.monotonic() - start_time
                remaining_budget = wait_budget_seconds - elapsed
                tile_animation_wait = max(0, min(animation_wait, remaining_budget))
                if tile_animation_wait > 0:
                    animation_wait_start = time.monotonic()
                    page.wait_for_timeout(tile_animation_wait * 1000)
                    animation_wait_elapsed = time.monotonic() - animation_wait_start

            # Per-tile timing breakdown so slow dashboards can be profiled from
            # logs alone. DEBUG rather than INFO: this fires once per tile, and
            # large dashboards can have dozens of tiles per report run.
            logger.debug(
                "Tile %s/%s timing: %.2fs waiting for spinners to clear, "
                "%.2fs waiting for animations.%s",
                i + 1,
                num_tiles,
                spinner_wait_elapsed,
                animation_wait_elapsed,
                context_suffix,
            )

            # Calculate what portion of the element we want to capture for this tile
            tile_start_in_element = i * tile_height
            remaining_content = dashboard_height - tile_start_in_element
            clip_height = min(tile_height, remaining_content)
            clip_y = (
                0
                if tile_height < remaining_content
                else tile_height - remaining_content
            )
            clip_x = dashboard_left

            # Skip tile if dimensions are invalid (width or height <= 0)
            # This can happen if element is completely scrolled out of viewport
            if clip_height <= 0 or clip_y < 0:
                logger.warning(
                    "Skipping tile %s/%s due to invalid clip dimensions: "
                    "x=%s, y=%s, width=%s, height=%s "
                    "(element may be scrolled out of viewport).%s",
                    i + 1,
                    num_tiles,
                    clip_x,
                    clip_y,
                    dashboard_width,
                    clip_height,
                    context_suffix,
                )
                continue

            # Clip to capture only the current tile portion of the element
            clip = {
                "x": clip_x,
                "y": clip_y,
                "width": dashboard_width,
                "height": clip_height,
            }

            # Take screenshot with clipping to capture only this tile's content
            tile_screenshot = page.screenshot(type="png", clip=clip)
            screenshot_tiles.append(tile_screenshot)

            logger.debug(
                "Captured tile %s/%s with clip %s%s",
                i + 1,
                num_tiles,
                clip,
                context_suffix,
            )

        # Combine all tiles
        logger.info("Combining screenshot tiles...%s", context_suffix)
        combined_screenshot = combine_screenshot_tiles(
            screenshot_tiles, log_context=log_context
        )

        return combined_screenshot

    except TiledScreenshotBudgetExceededError:
        # Budget exhaustion must fail cleanly, not be swallowed into a
        # `return None` (which upstream callers treat as "fall back to a
        # standard, unchecked screenshot").
        raise
    except Exception as e:
        logger.exception("Tiled screenshot failed: %s%s", e, context_suffix)
        return None
