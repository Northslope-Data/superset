/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
import { FeatureFlag, isFeatureEnabled } from '@superset-ui/core';
import { Grid } from '@superset-ui/core/components';

/**
 * Whether the mobile consumption-only experience is enabled for this
 * deployment. Non-hook variant for use inside styled-component
 * interpolations; prefer `useIsMobile` in components.
 */
export function isMobileConsumptionEnabled(): boolean {
  return isFeatureEnabled(FeatureFlag.MobileConsumptionMode);
}

/**
 * Returns true when the viewport is below antd's `md` breakpoint AND the
 * MOBILE_CONSUMPTION_MODE feature flag is enabled. All mobile-specific
 * behavior (route guarding, consumption-only chrome, drawer navigation)
 * should key off this hook so the flag remains a single kill switch.
 *
 * Defaults to desktop on the first render, before antd's
 * ResponsiveObserver has fired, so the initial paint never takes the
 * mobile branch by accident.
 */
export function useIsMobile(): boolean {
  const { md = true } = Grid.useBreakpoint();
  return !md && isMobileConsumptionEnabled();
}
