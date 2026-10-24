#pragma once
// The pixel face (2026-10-24): the look of the owner's site airwitech.com on the robot's square head - hand-drawn
// pixel glyphs on a dark screen, in the site's palette. Every emotion is composed of glyphs (eyes, brows, mouth,
// extras: tears, Zzz, hearts, a question mark, an anger mark, sparkles, sweat, sunglasses) on a grid of 32 x 24 square
// pixels of 10 px. Motion moves in whole pixels: the gaze shifts the eyes, blinks close them to a line, tears fall,
// Zzz rise, hearts beat, a new emotion dissolves in pixel by pixel. Plain C++ (no LVGL): the robot draws the grid
// through FaceEngine, tools/face_preview into a PNG.
//
// kPixelFace = false brings the smooth vector face (face_style.h) back.
#include <cstdint>

#include "face_style.h"

namespace face {

constexpr bool kPixelFace = true;
constexpr int kPixCols = 32, kPixRows = 24, kPixCell = kScreenW / kPixCols;  // 10 px

// The face's colour per emotion in the site's palette (paper, amber, cyan, violet, muted; coral for anger).
uint32_t pixelTone(Emotion e);

struct PixelGrid {
    uint32_t px[kPixRows][kPixCols];  // 0 = dark, else 0xRRGGBB | 0x1000000

    void clear();
    void put(int col, int row, uint32_t color);
    // Draws the lit pixels as squares of kPixCell - 1 px (a dark hairline between pixels); returns how many.
    int flush(Painter& out) const;
};

// One frame of an emotion with the animation state (gaze, blink, shake, phase, pulse as in face_style.h's Anim).
void renderPixelFace(Emotion e, const Anim& a, PixelGrid& out);

// A change of emotion: every pixel switches from `from` to `to` at its own moment of t 0..1 (a pixel dissolve).
void dissolve(const PixelGrid& from, const PixelGrid& to, float t, PixelGrid& out);

}  // namespace face
