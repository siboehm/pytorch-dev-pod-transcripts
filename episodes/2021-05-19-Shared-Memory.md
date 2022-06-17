---
layout: post
title: "Shared Memory"
date: 2021-05-19
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Shared Memory

Hello, everyone, and welcome to the Pritchard Tip Podcast.
Today, I wanna talk about a kind of niche topic, which actually you're probably using even though you don't know about it.
namely shared memory and Pytorch.
What is shared memory? Well, let's think about what happens on your computer when you want to run multiple processes.
Each process ordinarily has a separate memory adjust space.
that is isolated from every other process on your system.
And, you know, if you remember how your operating systems class explained it, there's, you know, a very fancy virtual memory system that your operating system implements along with your processor to actually make this possible.
So having your processes have separate memory is really good idea because, you know, you really don't want one process stomping over the memory of another process accidentally.
For example, you have a buggy, you know, Firefox instance, you don't want that to, you know, go into your bank account application.
That being said, sometimes it is useful to share some memory between two processes.
And your operating system also has a facility for that, and it's called shared memory.
Normally, shared memory gets used when you do shared libraries.
So what's the idea behind a shared library? Well, the idea behind a shared library is that you have a bunch of libraries on your system that might be used by multiple processes.
And it's a ways to actually, you know, have separate copies of exactly the same binary in each of the processes that you want.
So, you know, a shared library is designed in a way that one, it can be put anywhere in your dress space.
AKA, it is so called relocatable or it has been compiled with position independent code f pick as it's called And then, you know, the using the virtual address table, your operating system only needs to hold one copy of the shared memory and physical memory, and then we'll just, you know, map it to the various virtual address tables of all the processes that are actually using the shared library.
that's a really common use case I shared libraries in Unix like systems.
How about in Pytorch? Well, in Pytorch, shared memory can come in handy when you have a tensor, and you wanna share the contents between multiple processes.
Now, this is actually, you know, a little bit tricky to do.
Right? Because if you're wanting to write into the tensor, normally, if you have multiple concurrent, you know, processes or threads, working on writing something, you have to do some sort of synchronization.
But sort of, you know, one of the glorious things about machine learning is it doesn't really matter if you synchronize or not.
so called hog wild training methods actually worked pretty well.
And they just work by sort of, you know, yellowing the updates without any synchronization and things sort of just work out and they will end by the magic of gradient to sun.
So Pytorch has support for certain memory so that you can take a contents of a tensor and share it between multiple processes on a single machine.
And this is most useful usually because you know, Python is silly.
It's called the global interpreter lock.
So if you actually wanna do, you know, parallel processing on a single node, you usually need to have multiple processes to, like, actually max out your CPU.
Because otherwise, you're only gonna be running Python code on one core.
Okay.
So what does this look like in titles itself? Well, there's a few things that, you know, you have to know about shared memory that, like, lead to a bunch of things that Pytros does to sort of make this a seamless experience.
So one thing is that shared memory on your operating system is not reference counted.
In fact, once you create some shared memory, it will stay there indefinitely until someone explicitly decides they're gonna get rid of it.
And this kind of makes sense because, you know, shared memory is often represented as a file in a special dev shim mountain point on your operating system, like slash dev slash him.
And, you know, of course, files.
Files don't go away unless you actually arm them.
And so this leads to a problem which is that, you know, let's say that, you know, I allocate some shared memory.
Well, I need to get rid of it when I'm done with it.
Otherwise, it's gonna hang around until the end of my, you know, operating systems until it reboots or something like that.
So you could imagine setting up your process so that, you know, if the process, you know, is shutting down, then it can deallocate all the shared memory.
But this works out poorly if your process, for example, crashes for whatever reason, and none of the disruptors run-in that case.
So actually, Pytorch solves this problem by providing a sort of watchdog process.
This is the SHIM manager the shared memory manager.
And what the shared memory manager does is, you know, when we start using shared memory inside PyTorch, we spawn off a demonized version of this watch out process, whose only job in life is to watch the relevant processes that, you know, are associated with this Metroid instance, and when all of them are dead, clear all of the shared memory in question that it has been told about.
So in this particular case, shared memory, watch that process is much smaller.
It's not running custom user code.
It's just getting signals from the processes when shared memory is being allocated.
when it's being deallocated.
So it's much less likely to accidentally crash dual to a bug and, you know, it's a way we can make sure it shouldn't really actually know, gets reserved in this way.
Okay.
What are some other things that we need to do to make sure we're gonna work out? Well, another thing we need to do is we need to actually, you know, back our tensors with the shared memory in question.
So how does that work? Well, you know, we have a representation for tensor and you know, inside the tensor is a data pointer that points to some data.
And we represent this internally via a data pointer class, which sort of says, hey, here's where the data is, and also here's where to de out here's how to deallocate it.
And so the fact that the deallocateer for memory stored by tensors is actually, you know, user programmable means that you can actually override, you know, where things come from.
if you're just doing a normal tensor allocation, you just say, okay, I want the stock CPU allocator and that gives me a data pointer that says, okay, to free this memory just free it in the normal way.
But if you're doing share in memory and you wanna like pass it around with another process, then you can use a different allocator which says, okay, please allocate the shared memory for me.
And when it's done, deallocate it by, you know, both deallocate in the shared memory in whatever special way it needs to be.
And also, sending a message to the shared memory manager to say, okay, while I'm done with this January, you don't have to worry about it anymore.
And so in fact, the way we implement shared memory and Pytorch is there's actually a few allocators So there's a t h map allocator which says, okay, I'm just gonna give you some shared memory and then I'm gonna get rid of it, you know, un map it the normal way when you're done with it.
There's also a ref counted shared memory allocator, which says, okay.
Well, you can give me this shared memory, and I'll actually keep track of it via a ref count that is distributed over all Pytruch processes.
So you know, if I have multiple Pytorch processes that are referring to this shared memory, I won't deallocate it until the distributed rev count goes to zero.
And so once again, you know, what does the deallocator in this case do? Well, it just says, okay.
Well, when you're done, you know, decrement the distributed rev count and then also check if distributed Rev count has gone to zero, if so, free of the shared memory.
By the way, how the Rev counts are stored? Also shared memory.
And, you know, it's just The easiest way to implement this sort of thing.
And of course, the the managed shared memory allocator is the one that knows about the a shared memory manager, and that one does the stock behavior, but also talks to the shared memory manager to get things done.
Okay.
So that's it about shared memory on CPU, but it turns out that we also support shared memory on CUDA.
And the way we do that is sort of very similar.
CUDA API provides a way of taking some arbitrary CUDA memory and then saying, okay, create a opaque handle, some byte string, that when passed to another process, can be used to get another CUDA handle to the memory in question.
And so this way, you can also share CUDA memory across multiple processes.
How convenient? However, CUDA shared memory works a little differently than CPU shared memory.
Unlike CPU shared memory where, you know, if once you allocate it, it just stays live until you slowly delete it.
CUDA memory only stays live as long as the host process actually keeps the CUDA memory live.
And so for the longest time in titles, we had this restriction that, you know, when you have some CUDA shared memory, you must make sure that the CUDA shared memory stays live in the originating process long enough for all the, you know, consumer processes to be done using it Otherwise, very strange things will happen.
And, you know, these strange things include, you know, like, it being overwritten with total garbage because we remember we have caching allocator.
So we don't actually CUDA Maloc and CUDA free every time you allocate CUDA memory.
We, you know, allocate a big chunk of CUDA memory.
And then maybe sometime in the future, you know, we reuse it for something else.
So if someone else is still referring to, you know, some CUDA IBC memory that, you know, we decided was unused in the host side and then we used to something else, they'll see it actually get overwritten with some random data from the next allocation.
So that was a kind of foot gun.
And when Italy fed union joined the fighter jet project, one of the first things that he implemented was distributed rev counting for CUDA IPC sensors as well.
And it works kinda similar to how CPU rev counting works.
Right? So there is you know, shared memory file, hey, you know, shared memory once again, that maintains the distributed graph count.
And then there's just a sort of polling mechanism on the host side, which just looks and sees has the, you know, RevCon gone to zero, has the RevCon gone to zero, oh, the RevCon's gone to zero.
now I can release the tensor.
There were a bunch of different possibilities we had for how to go about doing this, but polling was the sort of simplest implement.
Okay.
So shared memory is a way to share memory between multiple processes in your system.
It's not so useful if you're doing multi node training but because Python has a Gill, it's pretty useful if you're using a single node and you just need multiple processes to paralyze.
You probably are using shared memory if you're using torch multi processing.
And there's just a few things that Hydrogen does to make this work out nicely.
But, you know, mostly, we're just relying on, you know, m map support for shared memory files.
So that's all I wanted to say today, talk to you next time.
.
