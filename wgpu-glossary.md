# Glossary of wgpu Terms

This was created solely as a learning exercise for myself and is intended for my own reference
whilst working on this project.  However, I assume I'm not the only person to be confused and
overwhelmed by the complexity of wgpu (and GPU programming in general) so this may be useful to you
as well.

- `Instance`: Singleton instance of wgpu.  Responsible for creating `Adapter`s and `Surface`s

- `Adapter`: Handle to a physical piece of silicon with which we can perform computation
- `Surface`: Handle to a region of the screen to which we can draw

- `Device`: Open connection to a compute device (i.e. an `Adapter`).  Performs computations by
  pushing `CommandBuffer`s to a `Queue` corresponding to this `Device`.  Creating any kind of
  GPU-accessible resource must occur through the corresponding `Device`.
- `Queue`: Handle to a command queue.  `CommandBuffer`s can be pushed to a queue (CPU-side), then
  are all submitted to the GPU (on `Queue::submit`) and executed asynchronously.

## GPU-Accessible Resources

- `Buffer`: Handle to a GPU-accessible region of memory, with no associated type.  A `Buffer` is
  either 'mapped' (the CPU can access/modify its contents) or 'unmapped' (the GPU can access its
  contents).  I believe these can't be done simultaneously.  Presumably, the data is sent to the GPU
  on the call to `Buffer::unmap`, and fetched on a call to `Buffer::map`?
- `Texture`: Handle to a GPU-accessible texture.  Like a `Buffer`, but always represents image data.
  - `TextureView`: TODO: Some equivalent to a reference to a `Texture` plus metadata?

## Executing Commands

- `CommandBuffer`: A sequence of (possibly related?) commands which are submitted to a `Queue` (with
  `Queue::submit`) then run.
- `CommandEncoder`: Factory type for 'recording' a sequence of GPU operations to be performed at a
  later date.  Created with `Device::create_command_encoder` and converted to a `CommandBuffer` with
  `CommandEncoder::finish`.

GPU commands can have the following types:

### Render Pass

A `RenderPass` is a recorded sequence of operations for rendering objects (e.g. frame rendering in
video games).  Created with `Encoder::begin_render_pass` and added to the `CommandBuffer` when
`drop`ped.

#### Attachments

Render passes take up two 'attachment's (outputs for rendered pixels).  One of these is a
`RenderPassColorAttachment` which receives the color data, and `RenderPassDepthStencilAttachment`
which receives both the depth and stencil channels.  Both of these take one `TextureView` as output
(except for multi-sampling where the color attachment takes an extra `TextureView`) and some
`Operations` to perform before and after rendering.  The `Operations` type is best described by its
fields:
- The `load` field describes what happens before rendering; either
  - `LoadOp::Clear`: write a fixed color to the buffer before rendering, or
  - `LoadOp::Load`: load the memory without doing modifying it.  TODO: Is this a no-op?
- The `store` field determines whether the rendered image is written back to the `Texture` after
  rendering.  I don't see any time where this wouldn't be set to `true`.  TODO: Surely there's a
  reason?

`Encoder::begin_render_pass` requires a `RenderPassDescriptor`, through which we give general
requirements for our  `RenderPass`.  This corresponds to the two 'attachments' (described above) and
a `Label` which is the name given to this `RenderPass`, used by e.g. GPU debuggers.
