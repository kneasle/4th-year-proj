## MT

The project is coming along pretty well - preliminary research and learning is mostly complete, and
code is starting to be written.  I'm starting by solving a simplified problem since doing so will
highlight any design issues without worrying about unnecessary complexity; instead of experimenting
with building part of a full image editor (which is full of small and uninteresting details that
take time to implement), I have decided to simplify things to an image with a single stack of
effects applied to one layer.  This 'prototype' version is looking to be working by the end of this
term or early next term.

## HT

The simplified prototype design from last term has borne fruit.  The whole purpose of this project
is to speed up image editing, and thus speed measurements are very important.  The measurements done
this term have exceeded my expectations, proving that the fundamental premise of the project is
sound (the premise being that performing image editing solely on the GPU is extremely fast because
transferring data is slow).  Therefore, what's left to do is to add more features so that the
project moves closer to a true image editor, making the measurements more applicable to the real
world.  I will likely do most of this over the vacation and write the report next term.
