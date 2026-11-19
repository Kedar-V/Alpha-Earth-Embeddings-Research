# Google Maps + Overture Overlay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Interactive Google Maps JS page with stratified Overture footprint overlays for 10 US metros.

**Architecture:** Build script samples buildings → GeoJSON + static HTML; serve script injects API key from `.env` via `/config.js`.

**Tech Stack:** Python (pandas, shapely), Google Maps JavaScript API, stdlib HTTP server.

**Spec:** `docs/superpowers/specs/2026-11-10-google-maps-overture-overlay-design.md`

## Task 1: Build script + HTML shell

- Create `scripts/build_google_maps_overture.py`
- Sample 3k 1:1 RES/NRES per metro, write GeoJSON + `regions.json` + `index.html` + `manifest.json` + README
- Run build; verify 10 geojson files exist

## Task 2: Serve script

- Create `scripts/serve_google_maps_overture.py`
- Serve `outputs/google_maps_overture/`; `/config.js` returns `window.GOOGLE_MAPS_API_KEY=...` from `.env`
- Smoke: curl index + config.js

## Task 3: End-to-end verify

- Start server on :8770; confirm HTML references Maps API and loads regions.json
