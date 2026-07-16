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

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from superset.utils.screenshot_utils import (
    _resolve_wait_budget_seconds,
    combine_screenshot_tiles,
    SCROLL_SETTLE_TIMEOUT_MS,
    take_tiled_screenshot,
    TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS,
    TiledScreenshotBudgetExceededError,
)


class TestCombineScreenshotTiles:
    def _create_test_image(self, width: int, height: int, color: str = "red") -> bytes:
        """Helper to create test PNG image bytes."""
        img = Image.new("RGB", (width, height), color)
        output = io.BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()

    def test_empty_tiles_returns_empty_bytes(self):
        """Test that empty tiles list returns empty bytes."""
        result = combine_screenshot_tiles([])
        assert result == b""

    def test_single_tile_returns_original(self):
        """Test that single tile returns the original image."""
        test_image = self._create_test_image(100, 100)
        result = combine_screenshot_tiles([test_image])
        assert result == test_image

    def test_combine_multiple_tiles_vertically(self):
        """Test combining multiple tiles into a single vertical image."""
        # Create test images with different colors
        tile1 = self._create_test_image(100, 50, "red")
        tile2 = self._create_test_image(100, 75, "green")
        tile3 = self._create_test_image(100, 25, "blue")

        result = combine_screenshot_tiles([tile1, tile2, tile3])

        # Verify result is not empty
        assert result != b""

        # Verify the combined image has correct dimensions
        combined_img = Image.open(io.BytesIO(result))
        assert combined_img.width == 100  # Max width of all tiles
        assert combined_img.height == 150  # Sum of all heights (50 + 75 + 25)

        # Verify the image format is PNG
        assert combined_img.format == "PNG"

    def test_combine_tiles_different_widths(self):
        """Test combining tiles with different widths uses max width."""
        tile1 = self._create_test_image(50, 100, "red")
        tile2 = self._create_test_image(150, 100, "green")
        tile3 = self._create_test_image(100, 100, "blue")

        result = combine_screenshot_tiles([tile1, tile2, tile3])

        combined_img = Image.open(io.BytesIO(result))
        assert combined_img.width == 150  # Max width
        assert combined_img.height == 300  # Sum of heights

    def test_combine_tiles_handles_pil_error(self):
        """Test that PIL errors are handled gracefully."""
        # Create one valid image and one invalid
        valid_tile = self._create_test_image(100, 100)
        invalid_tile = b"invalid_image_data"

        result = combine_screenshot_tiles([valid_tile, invalid_tile])

        # Should return the first (valid) tile as fallback
        assert result == valid_tile

    def test_combine_tiles_logs_exception(self):
        """Test that exceptions are logged properly."""
        with patch("superset.utils.screenshot_utils.logger") as mock_logger:
            # Create invalid image data that will cause PIL to raise an exception
            invalid_tile = b"definitely_not_an_image"
            valid_tile = self._create_test_image(100, 100)

            result = combine_screenshot_tiles([valid_tile, invalid_tile])

            # Should have logged the exception
            mock_logger.exception.assert_called_once()
            # Should return first tile as fallback
            assert result == valid_tile


class TestTakeTiledScreenshot:
    @pytest.fixture
    def mock_page(self):
        """Create a mock Playwright page object."""
        page = MagicMock()

        # Mock element locator
        element = MagicMock()
        page.locator.return_value = element

        # Mock element info - simulating a 5000px tall dashboard at position 100
        element_info = {"height": 5000, "top": 100, "left": 50, "width": 800}

        # Only one evaluate call needed for dashboard dimensions
        page.evaluate.return_value = element_info

        # Mock screenshot method
        fake_screenshot = b"fake_screenshot_data"
        page.screenshot.return_value = fake_screenshot

        return page

    def test_successful_tiled_screenshot(self, mock_page):
        """Test successful tiled screenshot generation."""
        with patch(
            "superset.utils.screenshot_utils.combine_screenshot_tiles"
        ) as mock_combine:
            mock_combine.return_value = b"combined_screenshot"

            result = take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

            # Should return combined screenshot
            assert result == b"combined_screenshot"

            # Should have called screenshot method multiple times
            # (3 tiles for 5000px height)
            assert mock_page.screenshot.call_count == 3

            # Should have called combine function
            mock_combine.assert_called_once()

    def test_element_not_found_returns_none(self):
        """Test that missing element returns None."""
        mock_page = MagicMock()
        element = MagicMock()
        element.wait_for.side_effect = Exception("Element not found")
        mock_page.locator.return_value = element

        result = take_tiled_screenshot(mock_page, "nonexistent", tile_height=2000)

        assert result is None

    def test_tile_calculation_logic(self, mock_page):
        """Test that tiles are calculated correctly."""
        # Mock dashboard height of 3500px with viewport of 2000px
        element_info = {"height": 3500, "top": 100, "left": 50, "width": 800}

        # Override the fixture's evaluate return for this test
        mock_page.evaluate.return_value = element_info

        with patch(
            "superset.utils.screenshot_utils.combine_screenshot_tiles"
        ) as mock_combine:
            mock_combine.return_value = b"combined"

            take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

            # Should take 2 screenshots (3500px / 2000px = 1.75, rounded up to 2)
            assert mock_page.screenshot.call_count == 2

    def test_logs_dashboard_info(self, mock_page):
        """Test that dashboard info is logged."""
        with patch("superset.utils.screenshot_utils.logger") as mock_logger:
            with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
                take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

                # Should log dashboard dimensions with lazy logging format
                mock_logger.info.assert_any_call(
                    "Dashboard: %sx%spx at (%s, %s)%s", 800, 5000, 50, 100, ""
                )
                # Should log number of tiles with lazy logging format
                mock_logger.info.assert_any_call("Taking %s screenshot tiles%s", 3, "")

    def test_exception_handling_returns_none(self):
        """Test that exceptions are handled and None is returned."""
        mock_page = MagicMock()
        mock_page.locator.side_effect = Exception("Unexpected error")

        with patch("superset.utils.screenshot_utils.logger") as mock_logger:
            result = take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

            assert result is None
            # The exception object is passed, not the string
            call_args = mock_logger.exception.call_args
            assert call_args[0][0] == "Tiled screenshot failed: %s%s"
            assert str(call_args[0][1]) == "Unexpected error"
            assert call_args[0][2] == ""

    def test_screenshot_clip_parameters(self, mock_page):
        """Test that screenshot clipping parameters are correct."""
        with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
            take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

            # Check screenshot calls have correct clip parameters
            screenshot_calls = mock_page.screenshot.call_args_list

            # Should have 3 tiles (5000px / 2000px = 2.5, rounded up to 3)
            assert len(screenshot_calls) == 3

            # All tiles use the same x and width
            for _, call in enumerate(screenshot_calls):
                kwargs = call[1]
                assert kwargs["type"] == "png"
                assert kwargs["clip"]["x"] == 50
                assert kwargs["clip"]["width"] == 800

            # Check y positions and heights for each tile
            # Tile 1: clip_y=0, height=2000 (tile_height < remaining: 5000)
            assert screenshot_calls[0][1]["clip"]["y"] == 0
            assert screenshot_calls[0][1]["clip"]["height"] == 2000

            # Tile 2: clip_y=0, height=2000 (tile_height < remaining: 3000)
            assert screenshot_calls[1][1]["clip"]["y"] == 0
            assert screenshot_calls[1][1]["clip"]["height"] == 2000

            # Tile 3: clip_y=1000 (tile_height - remaining: 2000 - 1000)
            # height=1000 (remaining content)
            assert screenshot_calls[2][1]["clip"]["y"] == 1000
            assert screenshot_calls[2][1]["clip"]["height"] == 1000

    def test_handles_invalid_tile_dimensions(self, mock_page):
        """Test that tiles with invalid dimensions are skipped."""
        # Mock a dashboard where the last tile would have 0 or negative height
        # This simulates edge cases in height calculations
        element_info = {"height": 4000, "top": 100, "left": 50, "width": 800}
        mock_page.evaluate.return_value = element_info

        with patch("superset.utils.screenshot_utils.logger") as mock_logger:
            with patch(
                "superset.utils.screenshot_utils.combine_screenshot_tiles"
            ) as mock_combine:
                mock_combine.return_value = b"combined"

                # Use exact viewport height that divides evenly
                result = take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

                # Should succeed
                assert result == b"combined"

                # Should take 2 screenshots (4000px / 2000px = 2)
                assert mock_page.screenshot.call_count == 2

                # Should not log any warnings about invalid dimensions
                warning_calls = [
                    call
                    for call in mock_logger.warning.call_args_list
                    if "invalid clip dimensions" in str(call)
                ]
                assert len(warning_calls) == 0

    def test_skips_tile_with_zero_height(self, mock_page):
        """Test that a tile with zero or negative height is skipped."""
        # This test verifies the clip_height <= 0 check
        # We'll manually test the logic by creating a scenario where
        # remaining_content becomes <= 0
        element_info = {"height": 2000, "top": 100, "left": 50, "width": 800}
        mock_page.evaluate.return_value = element_info

        with patch(
            "superset.utils.screenshot_utils.combine_screenshot_tiles"
        ) as mock_combine:
            mock_combine.return_value = b"combined"

            # Use viewport height equal to element height
            result = take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

            # Should succeed with 1 tile
            assert result == b"combined"
            assert mock_page.screenshot.call_count == 1

    def test_scroll_positions_calculated_correctly(self, mock_page):
        """Test that window scroll positions are calculated correctly."""
        with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
            take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

            # Check page.evaluate calls for scrolling
            # First call is for dimensions, subsequent are for scrolling
            evaluate_calls = mock_page.evaluate.call_args_list

            # Should have 1 dimension query + 3 scroll calls
            assert len(evaluate_calls) == 4

            # First call is for dimensions (contains querySelector)
            assert "querySelector" in str(evaluate_calls[0])

            # Subsequent calls are scroll positions
            # Tile 1: scroll to y=100 (dashboard_top + 0 * tile_height)
            assert evaluate_calls[1][0][0] == "window.scrollTo(0, 100)"

            # Tile 2: scroll to y=2100 (dashboard_top + 1 * tile_height)
            assert evaluate_calls[2][0][0] == "window.scrollTo(0, 2100)"

            # Tile 3: scroll to y=4100 (dashboard_top + 2 * tile_height)
            assert evaluate_calls[3][0][0] == "window.scrollTo(0, 4100)"

    def test_reset_scroll_position(self, mock_page):
        """Test that scroll position waits are called after each scroll."""
        with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
            take_tiled_screenshot(mock_page, "dashboard", tile_height=2000)

            # Should call wait_for_timeout 3 times (once per tile)
            assert mock_page.wait_for_timeout.call_count == 3

            # Each wait should use the scroll settle timeout constant
            for call in mock_page.wait_for_timeout.call_args_list:
                assert call[0][0] == SCROLL_SETTLE_TIMEOUT_MS

    def test_per_tile_spinner_wait_uses_viewport_check(self, mock_page):
        """wait_for_function polls viewport-visible spinners after each scroll."""
        with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
            take_tiled_screenshot(
                mock_page, "dashboard", tile_height=2000, load_wait=30
            )

        # 3 tiles → 3 wait_for_function calls, one per tile
        assert mock_page.wait_for_function.call_count == 3

        # Each call uses viewport-scoped JS and the load_wait timeout
        for call in mock_page.wait_for_function.call_args_list:
            js = call[0][0]
            assert "getBoundingClientRect" in js
            assert "window.innerHeight" in js
            assert call[1]["timeout"] == 30 * 1000

    def test_per_tile_spinner_timeout_logs_warning_and_continues(self, mock_page):
        """A per-tile spinner timeout logs a warning but still takes the screenshot."""
        from superset.utils.screenshot_utils import PlaywrightTimeout

        timeout = PlaywrightTimeout("mocked timeout")
        mock_page.wait_for_function.side_effect = timeout

        with patch("superset.utils.screenshot_utils.time.monotonic", return_value=0):
            with patch("superset.utils.screenshot_utils.logger") as mock_logger:
                with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
                    result = take_tiled_screenshot(
                        mock_page, "dashboard", tile_height=2000, load_wait=30
                    )

        # Screenshot should still proceed (non-fatal)
        assert result is not None
        # Warning (not error/exception) logged for each tile that timed out --
        # this is a customer chart-loading issue, not a Superset system fault.
        assert mock_logger.warning.call_count == 3
        assert mock_logger.error.call_count == 0
        mock_logger.warning.assert_any_call(
            "Timed out waiting for visible spinners to clear on tile "
            "%s/%s (waited %.1fs of a %ss requested load_wait; %.1fs "
            "elapsed of a %.1fs total budget; %s/%s tiles captured so "
            "far).%s",
            1,
            3,
            30,
            30,
            0.0,
            1440,
            0,
            3,
            "",
        )

    def test_load_wait_default_is_sixty_seconds(self):
        """load_wait defaults to 60 to match SCREENSHOT_LOAD_WAIT config default."""
        import inspect

        from superset.utils.screenshot_utils import take_tiled_screenshot

        sig = inspect.signature(take_tiled_screenshot)
        assert sig.parameters["load_wait"].default == 60

    def test_per_tile_animation_wait_called_per_tile(self, mock_page):
        """animation_wait adds an extra wait per tile after the spinner check."""
        with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
            take_tiled_screenshot(
                mock_page, "dashboard", tile_height=2000, animation_wait=5
            )

        # 3 tiles × (1 scroll settle + 1 animation wait) = 6 total calls
        assert mock_page.wait_for_timeout.call_count == 6

        animation_calls = [
            call
            for call in mock_page.wait_for_timeout.call_args_list
            if call[0][0] == 5 * 1000
        ]
        assert len(animation_calls) == 3

    def test_animation_wait_default_is_zero(self):
        """animation_wait defaults to 0 so no extra per-tile wait by default."""
        import inspect

        from superset.utils.screenshot_utils import take_tiled_screenshot

        sig = inspect.signature(take_tiled_screenshot)
        assert sig.parameters["animation_wait"].default == 0


class TestTileWaitBudget:
    @pytest.fixture
    def mock_page(self):
        """Create a mock Playwright page object for a 3-tile (5000px) dashboard."""
        page = MagicMock()
        element = MagicMock()
        page.locator.return_value = element
        page.evaluate.return_value = {
            "height": 5000,
            "top": 100,
            "left": 50,
            "width": 800,
        }
        page.screenshot.return_value = b"fake_screenshot_data"
        return page

    def test_per_tile_wait_shrinks_as_budget_depletes(self, mock_page, monkeypatch):
        """Each tile's spinner-wait timeout is capped at the remaining budget."""
        monkeypatch.setattr(
            "superset.utils.screenshot_utils.TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS",  # noqa: E501
            1000,
        )
        # monotonic() is called: once for start_time, then per tile once to
        # compute elapsed/remaining budget (before the spinner wait) and once
        # more right after the spinner wait (for the per-tile timing line).
        monotonic_values = iter([0, 0, 0, 950, 950, 990, 990])
        with patch(
            "superset.utils.screenshot_utils.time.monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
                result = take_tiled_screenshot(
                    mock_page, "dashboard", tile_height=2000, load_wait=100
                )

        assert result is not None
        timeouts = [
            call[1]["timeout"] for call in mock_page.wait_for_function.call_args_list
        ]
        # remaining budget: 1000, 50, 10 seconds -> capped timeouts shrink
        assert timeouts == [100 * 1000, 50 * 1000, 10 * 1000]
        assert timeouts == sorted(timeouts, reverse=True)

    def test_budget_exhausted_raises_and_stops_capturing(self, mock_page, monkeypatch):
        """Exhausting the budget aborts cleanly instead of capturing unchecked."""
        monkeypatch.setattr(
            "superset.utils.screenshot_utils.TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS",  # noqa: E501
            1000,
        )
        # start_time=0, tile 0: elapsed=0 (proceeds, captures, then a
        # post-spinner-wait timestamp for the timing line), tile 1 check:
        # elapsed=1000 -> remaining=0 -> raise before capturing.
        monotonic_values = iter([0, 0, 0, 1000])
        with patch(
            "superset.utils.screenshot_utils.time.monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            with patch(
                "superset.utils.screenshot_utils.combine_screenshot_tiles"
            ) as mock_combine:
                with patch("superset.utils.screenshot_utils.logger") as mock_logger:
                    with pytest.raises(TiledScreenshotBudgetExceededError):
                        take_tiled_screenshot(
                            mock_page, "dashboard", tile_height=2000, load_wait=100
                        )

        # Only the first tile was captured before the budget ran out.
        assert mock_page.screenshot.call_count == 1
        # Tiles were never combined -- the function raised before that point.
        mock_combine.assert_not_called()

        # Budget exhaustion is a customer chart-loading issue, not a Superset
        # system fault, so it must log at WARNING (not ERROR) -- consistent
        # with the #38130/#38441 precedent for screenshot timeout logging.
        assert mock_logger.error.call_count == 0
        mock_logger.warning.assert_called_once()
        warning_args = mock_logger.warning.call_args[0]
        assert "budget exhausted" in warning_args[0]
        # tile index, tiles total, tiles captured, tiles total,
        # elapsed seconds, budget seconds, log-context suffix
        assert warning_args[1] == 2
        assert warning_args[2] == 3
        assert warning_args[3] == 1
        assert warning_args[4] == 3
        assert warning_args[5] == 1000
        assert warning_args[6] == 1000
        assert warning_args[7] == ""

    def test_budget_exhausted_warning_includes_log_context(
        self, mock_page, monkeypatch
    ):
        """log_context (e.g. report execution id) is appended to the warning."""
        monkeypatch.setattr(
            "superset.utils.screenshot_utils.TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS",  # noqa: E501
            1000,
        )
        monotonic_values = iter([0, 0, 0, 1000])
        with patch(
            "superset.utils.screenshot_utils.time.monotonic",
            side_effect=lambda: next(monotonic_values),
        ):
            with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
                with patch("superset.utils.screenshot_utils.logger") as mock_logger:
                    with pytest.raises(TiledScreenshotBudgetExceededError):
                        take_tiled_screenshot(
                            mock_page,
                            "dashboard",
                            tile_height=2000,
                            load_wait=100,
                            log_context="execution_id=abc-123",
                        )

        warning_args = mock_logger.warning.call_args[0]
        assert warning_args[-1] == " [execution_id=abc-123]"

    def test_per_tile_timing_debug_line_logged(self, mock_page):
        """Each tile logs a DEBUG timing breakdown (spinner wait, animation wait)."""
        with patch("superset.utils.screenshot_utils.logger") as mock_logger:
            with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
                take_tiled_screenshot(
                    mock_page,
                    "dashboard",
                    tile_height=2000,
                    log_context="cache_key=xyz",
                )

        timing_calls = [
            call for call in mock_logger.debug.call_args_list if "timing" in call[0][0]
        ]
        assert len(timing_calls) == 3
        for i, call in enumerate(timing_calls):
            args = call[0]
            assert args[1] == i + 1  # tile index
            assert args[2] == 3  # total tiles
            assert args[-1] == " [cache_key=xyz]"

    def test_fast_dashboard_matches_default_behavior(self, mock_page):
        """Well under budget, waits are not capped and behavior is unchanged."""
        with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
            result = take_tiled_screenshot(
                mock_page,
                "dashboard",
                tile_height=2000,
                load_wait=30,
                animation_wait=5,
            )

        assert result is not None
        assert mock_page.screenshot.call_count == 3

        for call in mock_page.wait_for_function.call_args_list:
            assert call[1]["timeout"] == 30 * 1000

        animation_calls = [
            call
            for call in mock_page.wait_for_timeout.call_args_list
            if call[0][0] == 5 * 1000
        ]
        assert len(animation_calls) == 3


class TestResolveWaitBudgetSeconds:
    """The budget is derived from the running Celery task's own time limit
    when available, and falls back to the fixed constant otherwise."""

    def _mock_task(self, soft=None, hard=None):
        task = MagicMock()
        task.request.timelimit = (soft, hard)
        return task

    def test_derives_budget_from_soft_time_limit(self):
        """soft_time_limit is preferred over the hard time_limit when both are set."""
        task = self._mock_task(soft=90, hard=120)
        with patch("superset.utils.screenshot_utils.current_task", task):
            budget = _resolve_wait_budget_seconds()

        # margin = min(300, 90 * 0.2) = 18; budget = 90 - 18 = 72
        assert budget == 72

    def test_small_task_limit_yields_positive_scaled_margin_budget(self):
        """A 120s thumbnail-task limit (superset-shell#4389) still gets a
        usable, positive budget via the scaled-down margin, not the fixed
        300s margin that would otherwise wipe it out."""
        task = self._mock_task(soft=None, hard=120)
        with patch("superset.utils.screenshot_utils.current_task", task):
            budget = _resolve_wait_budget_seconds()

        # margin = min(300, 120 * 0.2) = 24; budget = 120 - 24 = 96
        assert budget == 96
        assert budget > 0
        assert budget < 120

    def test_no_task_context_falls_back_to_constant(self):
        """Outside of a Celery task, the fixed fallback budget is used."""
        with patch("superset.utils.screenshot_utils.current_task", None):
            budget = _resolve_wait_budget_seconds()

        assert budget == TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS

    def test_task_with_no_timelimit_falls_back_to_constant(self):
        """A task with no soft or hard limit set falls back to the constant."""
        task = self._mock_task(soft=None, hard=None)
        with patch("superset.utils.screenshot_utils.current_task", task):
            budget = _resolve_wait_budget_seconds()

        assert budget == TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS

    def test_derivation_exception_falls_back_to_constant(self):
        """Any failure while inspecting the task context must never break a
        screenshot -- fall back to the constant and log at DEBUG."""

        class _BrokenTask:
            """Simulates a task-like object whose .request raises."""

            @property
            def request(self):
                raise RuntimeError("boom")

        with patch("superset.utils.screenshot_utils.current_task", _BrokenTask()):
            with patch("superset.utils.screenshot_utils.logger") as mock_logger:
                budget = _resolve_wait_budget_seconds(log_context="execution_id=abc")

        assert budget == TILED_SCREENSHOT_TOTAL_WAIT_BUDGET_SECONDS
        mock_logger.debug.assert_called_once()
        debug_args = mock_logger.debug.call_args
        assert "Failed to derive" in debug_args[0][0]
        assert debug_args[1]["exc_info"] is True

    def test_take_tiled_screenshot_uses_derived_budget_from_task_limit(self):
        """take_tiled_screenshot caps waits using the task-derived budget."""
        mock_page = MagicMock()
        mock_page.locator.return_value = MagicMock()
        mock_page.evaluate.return_value = {
            "height": 5000,
            "top": 100,
            "left": 50,
            "width": 800,
        }
        mock_page.screenshot.return_value = b"fake_screenshot_data"

        task = self._mock_task(soft=None, hard=120)
        with patch("superset.utils.screenshot_utils.current_task", task):
            with patch("superset.utils.screenshot_utils.combine_screenshot_tiles"):
                take_tiled_screenshot(
                    mock_page, "dashboard", tile_height=2000, load_wait=200
                )

        # load_wait=200s requested, but the derived 96s budget caps the very
        # first tile's wait well below that (allow a small tolerance for the
        # real wall-clock time elapsed between deriving the budget and
        # capping the first tile's wait).
        first_timeout = mock_page.wait_for_function.call_args_list[0][1]["timeout"]
        assert first_timeout == pytest.approx(96 * 1000, abs=1000)
        assert first_timeout < 200 * 1000
