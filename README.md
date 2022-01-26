# 4th Year Project

Note: Currently, this project is unnamed.  Once I've worked on it more, I'll give it a name.

This is a project to create a fully GPU-accelerated non-destructive image editor.  Technically, the
goal for specifically my 4th year is to implement a 'back-end' library that can then be used to
build a full image editor.

'Destructive' editing means that adding effects (e.g. colour adjustment, rotations, scaling, etc.)
will permanently overwrite the underlying layer.  GIMP's filters are destructive, and this is
frustrating to the point of starting my own image editor.  Conversely, 'non-destructive' effects
don't destroy the underlying layer - you can apply any numbers of effects and still modify the old
ones.

This project is fully GPU-accelerated.  This means that all images are stored and processed on the
graphics processor, and never copied between processors unless you're saving a file (in which case,
copying can't be avoided).  This _should_ be insanely fast.  The goal is that the user should never
have to wait for the image to update - adjusting effect settings should always update, in real time,
hopefully at 60fps.
