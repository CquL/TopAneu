---
name: TopAneu technical diagnostic
description: Read-only technical report for model error diagnosis and experiment prioritization.
colors: bg/primary bg/secondary bg/tertiary border/light border/default text/primary text/secondary text/tertiary icon/accent blue/500 green/700 purple/500
typography: System Sans Variable, text/xs/normal, text/xs/semibold, text/sm/normal, text/sm/medium, heading/md/medium, heading/2xl
spacing: space-12, space-24
rounded: corner-radius/cr-24
surfaces: 1140px desktop, 800px content, 48px top bar, 24px section gap
components: top-bar, metric-card, report-block, chart-block, table-list, popover-menu
implementation: canonical portable analytics artifact reader
---

## Overview

Answer-first technical report with compact evidence and explicit definitions.

## Colors

Use neutral surfaces and one blue root for quantitative marks; green/700 and purple/500 remain reserved tokens and are not required for categorical decoration.

## Typography

Use the declared System Sans Variable hierarchy and mono-aligned numbers where supported.

## Layout

Use full-width reading order, responsive metric cards, and a single 1140px report canvas with 800px narrative measure.

## Elevation & Depth

Use elevation/01 only for report cards; avoid extra shells around charts.

## Shapes

Use corner-radius/cr-24 consistently; do not add decorative geometry.

## Components

The canonical reader owns top-bar, metric-card, report-block, chart-block, table-list, and popover-menu behavior.

## Do's and Don'ts

Do preserve denominators and source details. Do not imply causal certainty from descriptive OOF diagnostics. Users can customize narrative text, layout order, and chart presentation through the canonical artifact workflow.
