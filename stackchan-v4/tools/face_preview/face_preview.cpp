// Draws every emotion of main/ui/face_style.cpp into one PPM image (a grid of
// 320x240 screens) with a tiny anti-aliased software painter - the same
// renderFace() the robot uses, without LVGL. Build/run: tools/face_preview/run.sh
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../main/ui/face_style.h"
#include "../../main/ui/pixel_face.h"

namespace {

constexpr int kSS = 3;  // supersampling per axis

struct Canvas : face::Painter {
    int w, h;
    std::vector<float> rgb;  // supersampled
    Canvas(int w_, int h_) : w(w_ * kSS), h(h_ * kSS), rgb(static_cast<size_t>(w_ * kSS * h_ * kSS * 3)) {}
    void clear(uint32_t c)
    {
        for (int i = 0; i < w * h; ++i) setPix(i % w, i / w, c, 1.0f);
    }
    void setPix(int x, int y, uint32_t c, float a)
    {
        if (x < 0 || y < 0 || x >= w || y >= h) return;
        float* p = &rgb[static_cast<size_t>((y * w + x) * 3)];
        const float cr = ((c >> 16) & 0xFF) / 255.0f, cg = ((c >> 8) & 0xFF) / 255.0f, cb = (c & 0xFF) / 255.0f;
        p[0] = p[0] * (1 - a) + cr * a;
        p[1] = p[1] * (1 - a) + cg * a;
        p[2] = p[2] * (1 - a) + cb * a;
    }
    template <class F> void fill(float x1, float y1, float x2, float y2, uint32_t c, float a, F inside)
    {
        const int X1 = std::max(0, static_cast<int>(std::floor(x1 * kSS))), X2 = std::min(w - 1, static_cast<int>(std::ceil(x2 * kSS)));
        const int Y1 = std::max(0, static_cast<int>(std::floor(y1 * kSS))), Y2 = std::min(h - 1, static_cast<int>(std::ceil(y2 * kSS)));
        for (int y = Y1; y <= Y2; ++y)
            for (int x = X1; x <= X2; ++x)
                if (inside((x + 0.5f) / kSS, (y + 0.5f) / kSS)) setPix(x, y, c, a);
    }
    void roundRect(int x1, int y1, int x2, int y2, int radius, uint32_t color, uint8_t opa) override
    {
        const float r = std::min<float>(radius, std::min(x2 - x1, y2 - y1) / 2.0f);
        const float X1 = static_cast<float>(x1), Y1 = static_cast<float>(y1), X2 = x2 + 1.0f, Y2 = y2 + 1.0f;
        fill(X1, Y1, X2, Y2, color, opa / 255.0f, [&](float x, float y) {
            if (x < X1 || x > X2 || y < Y1 || y > Y2) return false;
            const float cx = std::clamp(x, X1 + r, X2 - r), cy = std::clamp(y, Y1 + r, Y2 - r);
            return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r;
        });
    }
    void circle(int cx, int cy, int r, uint32_t color, uint8_t opa) override
    {
        fill(cx - r - 1.0f, cy - r - 1.0f, cx + r + 1.0f, cy + r + 1.0f, color, opa / 255.0f,
             [&](float x, float y) { return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= float(r * r); });
    }
    void triangle(int x1, int y1, int x2, int y2, int x3, int y3, uint32_t color) override
    {
        auto edge = [](float ax, float ay, float bx, float by, float px, float py) {
            return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
        };
        const float area = edge(x1, y1, x2, y2, x3, y3);
        fill(std::min({x1, x2, x3}) - 1.0f, std::min({y1, y2, y3}) - 1.0f, std::max({x1, x2, x3}) + 1.0f,
             std::max({y1, y2, y3}) + 1.0f, color, 1.0f, [&](float x, float y) {
                 const float e1 = edge(x1, y1, x2, y2, x, y), e2 = edge(x2, y2, x3, y3, x, y), e3 = edge(x3, y3, x1, y1, x, y);
                 return area >= 0 ? (e1 >= 0 && e2 >= 0 && e3 >= 0) : (e1 <= 0 && e2 <= 0 && e3 <= 0);
             });
    }
    void line(int x1, int y1, int x2, int y2, int width, uint32_t color, uint8_t opa) override
    {
        const float r = width / 2.0f;
        const float dx = x2 - x1, dy = y2 - y1, len2 = dx * dx + dy * dy;
        fill(std::min(x1, x2) - r - 1, std::min(y1, y2) - r - 1, std::max(x1, x2) + r + 1, std::max(y1, y2) + r + 1, color,
             opa / 255.0f, [&](float x, float y) {
                 float t = len2 > 0 ? ((x - x1) * dx + (y - y1) * dy) / len2 : 0;
                 t = std::clamp(t, 0.0f, 1.0f);
                 const float px = x1 + t * dx - x, py = y1 + t * dy - y;
                 return px * px + py * py <= r * r;
             });
    }
    void arc(int cx, int cy, int r, int width, int startDeg, int endDeg, uint32_t color) override
    {
        const float ro = static_cast<float>(r), ri = static_cast<float>(r - width);
        const float half = width / 2.0f, rm = (ro + ri) / 2;
        auto norm = [](float d) {
            while (d < 0) d += 360;
            while (d >= 360) d -= 360;
            return d;
        };
        const float s = norm(static_cast<float>(startDeg)), span = static_cast<float>(endDeg - startDeg);
        const float sr = startDeg * 3.14159265f / 180, er = endDeg * 3.14159265f / 180;
        const float sx = cx + rm * std::cos(sr), sy = cy + rm * std::sin(sr), ex = cx + rm * std::cos(er), ey = cy + rm * std::sin(er);
        fill(cx - ro - 1.0f, cy - ro - 1.0f, cx + ro + 1.0f, cy + ro + 1.0f, color, 1.0f, [&](float x, float y) {
            const float d = std::hypot(x - cx, y - cy);
            const float ang = norm(std::atan2(y - cy, x - cx) * 180.0f / 3.14159265f);
            const bool in_span = span >= 360 || norm(ang - s) <= span;
            if (d >= ri && d <= ro && in_span) return true;
            return std::hypot(x - sx, y - sy) <= half || std::hypot(x - ex, y - ey) <= half;  // rounded ends
        });
    }
};

}  // namespace

int main(int argc, char** argv)
{
    const char* out = argc > 1 ? argv[1] : "faces.ppm";
    // argv[2]: "vector" draws the smooth vector faces instead of the pixel ones
    const bool pixel = face::kPixelFace && !(argc > 2 && std::strcmp(argv[2], "vector") == 0);
    const int cols = 7, rows = (kEmotionCount + cols - 1) / cols;
    const int W = face::kScreenW, H = face::kScreenH, gap = 6;
    const int OW = cols * (W + gap), OH = rows * (H + gap);
    std::vector<uint8_t> img(static_cast<size_t>(OW * OH * 3), 40);
    for (int e = 0; e < kEmotionCount; ++e) {
        Canvas c(W, H);
        c.clear(face::kBackground);
        face::Anim a;
        a.pulse = 0.5f;
        a.phase = 0.35f;
        if (pixel) {  // as FaceEngine draws it
            a.pulse = 0.8f;
            static face::PixelGrid grid;
            face::renderPixelFace(static_cast<Emotion>(e), a, grid);
            grid.flush(c);
        } else {
            face::renderFace(face::shapeFor(static_cast<Emotion>(e)), a, c);
        }
        const int gx = (e % cols) * (W + gap), gy = (e / cols) * (H + gap);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int k = 0; k < 3; ++k) {
                    float s = 0;
                    for (int sy = 0; sy < kSS; ++sy)
                        for (int sx = 0; sx < kSS; ++sx) s += c.rgb[static_cast<size_t>(((y * kSS + sy) * c.w + x * kSS + sx) * 3 + k)];
                    img[static_cast<size_t>(((gy + y) * OW + gx + x) * 3 + k)] = static_cast<uint8_t>(std::clamp(s / (kSS * kSS), 0.0f, 1.0f) * 255);
                }
    }
    FILE* f = std::fopen(out, "wb");
    if (!f) return 1;
    std::fprintf(f, "P6\n%d %d\n255\n", OW, OH);
    std::fwrite(img.data(), 1, img.size(), f);
    std::fclose(f);
    for (int e = 0; e < kEmotionCount; ++e) std::printf("%d:%s ", e, emotionName(static_cast<Emotion>(e)));
    std::printf("\n");
    return 0;
}
