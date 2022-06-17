---
layout: post
title: "Asynchronous Versus Synchronous Execution"
date: 2021-07-26
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Asynchronous Versus Synchronous Execution

Hello, everyone, and welcome to the Pytorch Dev podcast.
Today, I want to talk a little bit more about blocking versus unblocking APIs in Pytorch.
and its implications on various design questions in Pytorch.
In the CUDA podcast that I gave a long time ago, I mentioned how CUDA is asynchronous.
That doesn't say, when you do operations on CUDA tensors, they don't execute immediately.
The Program will actually return from the function you call before it's actually done during the operation question.
Instead, in the background, your GPU will be Chugging away, doing the IO computation in question, and your actual Python program is allowed to run ahead and figure out what the next thing that it needs to do in order to execute the next operation is.
This is in contrast to CPU execution, which is synchronous.
So when you ask for a CPU operation on something, while you're going to wait until the CPU operation is entirely done before moving on to the next thing in question.
Asynchronous execution for CUDA is pretty nice because it means that we aren't bottlenecked by the Python interpreter overhead so long as we queue enough work for the GPU to do.
You just have to wait until you've started up actually doing commutation.
And then any further overhead from the Python side program can be covered up as long as, you know, you've got enough work to do because you'll probably hit and queue the next piece of work before the GPU is actually ready to do it.
On the same on the flip side of the coin, it's nice for CPU to be synchronous because well, it means that, you know, once you actually have a CPU denser, it's actually got the honest goodness data.
if you wanna say FFI it out to some other system by passing on the raw data pointer, there's nothing special you have to do.
It just works.
And of course, it's a lot easier to implement asynchronous API than an asynchronous one.
because then you have to decide all sorts of questions about, you know, how exactly you're going to notify the threads that something's ready, how exactly you're gonna, like, queue things to execute.
And all in all, it just removes a bunch of implementation complexities that you have to deal with.
By the way, CUDA by default polls, but, you know, you can actually change out how exactly it does the synchronization between the thread that's actually executing things and your native thread by toggling my social configuration in the CUDA API.
So both of these paradigms make sense when you operate exclusively in CPU or exclusively in CUDA, you know, there isn't too much to worry about but there are a bunch of places in our API where we interact between CPU and CUDA, and this is the point at which it actually is a little non trivial to deal with the impedance matching between these two paradigms.
So to look at one particular example, let's look at the non blocking argument on the two method on tensor.
So what does this do? Well, it says when normally we have a Conversion from a CPU Tensor to a CUDA Tensor or vice versa, we will wait until the conversion is completely done before returning from this function.
And non blocking says, actually, don't bother waiting.
Just go ahead and return immediately while the, you know, CUDA driver is doing this asynchronous update.
And let me go ahead and do other things in the Python program.
So it doesn't take too long to realize why we don't default to non blocking execution by default.
Let's think about the CPU to CUDA case.
So the CPU to CUDA case is not such a big deal.
Right? So you have some memory, and you wanna transfer it to CUDA.
And, you know, like, your CUDA kernels are already gonna be asynchronously executed after this particular host a device copy happens.
So what's the big deal? Well, there are two problems.
One is that when CUDA does memory transfer, it needs to actually have the memory in some location so that the GPU hardware can actually direct the memory access it out of the RAM in your actual CPU.
And so to do that, you need some special memory called page locked memory.
And the way you get that is using a pin memory allocator.
in PyTorch.
That's from the CUDA API.
So you can't do non blocking CPU to CUDA or vice versa operations by default, you actually need your CPU tensors to live in pinned memory.
And pinned memory isn't free because like, when you say when you pin the memory, you're saying to the operating system, you're not allowed to, like, move it to swap, you're not allowed to move it around.
And so it reduces the amount of flexibility your operating system has to deal with your CPU memory.
So by default, PyTorch doesn't allocate pin memory.
By the way, Cafe two did allocate pin memory by default, but Pritchard doesn't do that.
And so you need to make sure you like ahead of time actually pin things if you wanna use non blocking.
But that's not even the end of your troubles.
So if you do a CPU to CUDA operation on panel memory, you will have some, you know, thread in the CUDA runtime going ahead and copying the data from CPU to CUDA.
What happens if someone goes ahead and mutates that CPU tester while this transfer is taking place.
Well, you'll get nonsense in this two because it's not like we went ahead and made a copy of the CPU buffer before we did the transfer.
Right? The whole point of non blocking is to make things run faster.
And, you know, the way it actually makes things not faster in this particular case with pin memory is we get to avoid actually having to do a copy into pin memory before we do the operation in question.
So we're reading directly out of the source sensor zero copies and that means you actually to make sure that the sensor six around, doesn't get deallocated, doesn't get overwritten until you're done doing this memory transfer.
And of course, ordinarily, it would be safe to override the CPU Tensor immediately after the two operation returns except remember you said it was non blocking.
Right? So it's gonna return immediately regardless of whether or not the copy is finished or not.
The reverse situation is even worse.
So when you have CUDA going to CPU, ordinarily, you know, once again, this will block until everything has been copied into CPU.
If you specify that to be non blocking, then we will immediately return.
We will have given you a CPU tensor, but the CPU tensor will be filled with garbage until some undeterminate time in the future when the device to host copy finishes.
And in fact, the only way to properly wait for this transfer to finish is to either do a CUDA synchronized, which is just a blocking operation waiting for everything in, you know, CUDA to make its way back to CPU, or if you wanna be a little more fine green about it because you're running multiple streams or multiple other types of concurrency, you can set up an event on CUDA, which will trigger after this copy is done.
So there are a lot of caveats here and it is not easy to use these APIs correctly, but, you know, one of the philosophies of PyTorch is right like give a simple API, not a easy to use one necessarily.
And so we give people all the tools they need we have reasonably simple semantics.
And in this case, you know, you're kinda just up to your own to make sure you do everything correctly.
And there is a performance to be gained here so people will use non blocking to get that performance in the situation.
There's been a long standing idea running around that no one has implemented yet to sort of make this situation a little better, and it's called async CPU.
Right? So I talked about how CPU is synchronous, and one of the reasons why it criticists.
It's just easier and more efficient to implement because you don't need any blocking mechanisms.
But there's nothing stopping us from having a CUDA like asynchronous execution model, except all the execution is happening on CPU.
So we dubbed this async CPU.
The idea behind async CPU is it would be a different device distinct from the CPU device.
You would share all the kernels that regular Cypher uses But when you do an operation instead of immediately going ahead and running the CPU computation to the end, we would, you know, put this in some sort of queue for some worker thread to actually execute the actual operation on.
And once again the idea is, you know, if you have multiple threads, and and you have a lot of work to do.
You may be able to successfully have the control thread run ahead and you know, make up for the fixed overhead of doing all the synchronizes correctly in this multi threaded concept context to avoid, you know, once again, cover up the latency from executing Python programs.
An added benefit, which is, you know, sort of drawing from the discussion we just had is if we had an async CPU tensor, we could give a user friendly API for Cruded's CPU non blocking copies.
Right? So what you would do is you would say, CUDA CPU doesn't return a CPU Tensor.
It returns an async CPU Tensor.
and you can now just directly run operations on it and rest assured that you those operations would only ever actually execute once the device to host copy had actually finished.
So the async CPU idea has been around for a long time And for the longest time, we never implement it, and there was a good reason why we didn't implement it.
Right? Which is that adding a new device to my church is a lot of work.
Right? We've got so many operators.
And, you know, if you had a new device like using CPU, well, yes, you can reuse all of the kernels that you, you know, had for the CPU thing, but, you know, async somehow.
You still have to actually handle computing the metadata for the tensor you are going to return from the async CPU operation.
To explain this in more detail, it's useful thinking about what are blocking versus non blocking operations on CUDA tensors.
So we've already established that doing something like a device to host transfer aka what would happen if you called say item on a CUDA Tensor is blocking.
Right? We have to wait until we get the actual data in a CPU before we can do anything with it.
But there are a lot of also methods on tensors which are not blocking.
For example, I can take a CUDA Tensor and I can ask for what its sizes and this doesn't actually cause us to synchronize with the GPU waiting for all the operations to finish.
Why? Well, it's because the size information is maintained on CPU.
Right? It's not something that's stored in CUDA, it's stored on CPU.
Many things are like this.
In fact, you know, if I ask you a question like, hey, Here are two CUDA tensors.
Do they overlap in memory? Well, I don't need to actually do a synchronizer with CUDA because I have my CUDA data pointers and I can just look at those and the sizes and the strides and figure out if there's a overlap or not.
So the problem with a sync CPU, right, is that whenever you wanna do an async back end, you need to actually say what the output like size and strides and everything else's without actually running the kernel in question.
And that would have been a lot of work.
You would have to do it for every operator and so no one really wanted to do the work and so async CPU never became a thing.
Fortunately, there's a project called meta tensors, which allows you to run the operations without doing the computation question and figure out what the output tensor size, d type, everything like that looks like.
So basically, assuming that you have something that is like meta tensors, you actually basically have most of the pieces you need for doing asynchronous CPU generation.
And you just need to like sick a code gen on the problem to generate a fast unboxed kernels that, like, put the arguments on the queue and send ship them off wherever else.
to actually execute.
So async CPU is a project that probably finally has gotten its time, but with meta tensors.
and, you know, it just needs someone to actually go ahead and work on it.
Stuff gets really weird when you're in the asynchronous world, though.
So I wanna give one more example of non blocking, making things very complicated, and that's in the CUDA caching allocator.
So the CUDA caching allocator is a way of, you know, allocating CUDA memory without actually hitting CUDA Malek, which in old versions of CUDA was very slow.
So we maintain this big pile of CUDA memory.
And, you know, when you ask for an allocation, we look in it, find a free spot that, you know, has enough space and we give that to you.
And similarly, if you give us back some memory, you freeze some memory, we just return it to the pool so someone else can use it.
So the hazard in the CUDA Cashing allocator is what happens if someone returns some memory to the CUDA Cashing allocator, which By the way, this is entirely CPU side.
There's no synchronization involved.
And then the CUDA caching allocator goes ahead and hands out the memory to someone else But at the same time, you are still executing the asynchronous kudos that were expecting the CUDA memory to be live.
So you're in one of these very awkward situations where I can have some CUDA memory in the CUDA caching allocator According to the state on CPU, it is free.
But actually, we are still operating on that memory in a bunch of backlog async CUDA kernels that are coding.
Oof.
Now, ordinarily, this doesn't cause any problems because remember, CUDA is organized into these streams So whatever up if you have only operating on a single stream, well, if you say, okay, now I'm gonna reallocate this memory for someone else and trash it.
That trashing operation happens in the stream and will happen after all the original CUDA colonals that were, you know, waiting to work on the original data before you get there.
So, you know, the race is averted.
But that's only true if all those operations are on the same stream.
And as I said, we support multiple streams in Pytorch, and so you can actually end up with the data showing up on a different stream, you know, and then there's no guarantee synchronization.
So to handle that, we, you know, force people to also record stream information when they run their kernels, and this is how we insert the necessary events to make sure that we actually go ahead and wait for all those informations to be done before the caching allocator you know, you can actually use this memory that you've got in from the CUDA caching allocator.
Okay.
So that's been a whirlwind tour of async and synchronous execution and how to put them together.
That's everything I wanted to say for today.
Talk to you next time.
.
