/*
 * On-screen controls for a touch screen (an iPad's, an Android tablet's): a
 * DualShock 2 drawn over the picture and read as the window's controller when
 * there is no other.
 *
 *   - the left stick: a finger put down in the lower left of the screen is its
 *     centre, and where it moves is the stick's (a floating stick);
 *   - Cross, Circle, Square and Triangle in the lower right, a finger sliding
 *     from one to the next as on a pad; L1 and R1 above them; START at the top.
 *
 * A finger put down anywhere else is the mouse's pointer (the port's menus take
 * it: a tap is a click). The controls fade out when a game controller is used
 * and come back at the next touch.
 *
 * Window pixels throughout; the window's thread feeds the fingers, the game's
 * thread reads the pad (Merge) and the presenting thread the shapes (Shapes):
 * a lock between them.
 */

#ifndef PS2GPU_TOUCH_CONTROLS_H
#define PS2GPU_TOUCH_CONTROLS_H

#include <cstdint>
#include <mutex>
#include <vector>

namespace ps2gpu {

class TouchControls {
public:
    /* What is drawn: a round control at (x, y) of radius r, its kind (Shape
       below) and whether it is held; the shader draws each. */
    enum Kind : std::uint32_t { kStickBase, kStickKnob, kCross, kCircle, kSquare, kTriangle, kL1, kR1, kStart };
    struct Shape {
        float x, y, r, alpha;
        float kind, held, pad0, pad1;
    };

    /* A finger: down (true if a control took it, false if it is the
       pointer's), moved, up. Positions in window pixels; the window's size. */
    bool Down(std::int64_t finger, float x, float y, float width, float height);
    void Move(std::int64_t finger, float x, float y, float width, float height);
    void Up(std::int64_t finger);
    /* A controller was used: the controls fade out (until the next touch). */
    void ControllerUsed();

    bool Visible() const;
    /* The held controls into a DualShock 2's state (display.h's PadState
       order): true if any finger is on a control. */
    bool Merge(std::uint16_t* buttons, std::uint8_t sticks[4]) const;
    std::vector<Shape> Shapes(float width, float height) const;

private:
    struct Place { Kind kind; float x, y, r; };
    static std::vector<Place> Layout(float width, float height);
    static bool InStickZone(float x, float y, float width, float height);
    int ButtonAt(float x, float y, float width, float height) const;     // -1: none

    struct Finger {
        std::int64_t id;
        int control;            // a Kind (kStickBase for the stick), or -1: the pointer's
        float x0, y0;           // where the stick's finger came down: its centre
        float x, y;
    };
    mutable std::mutex lock_;
    std::vector<Finger> fingers_;
    bool visible_ = false;
    float width_ = 0, height_ = 0;
};

}  // namespace ps2gpu

#endif
