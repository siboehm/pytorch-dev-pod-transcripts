---
layout: post
title: "Torchdeploy"
date: 2021-06-09
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Torchdeploy

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk about torch deploy, a way of deploying direct Python code in production environments where you can't wait for the Gil.
So what is torch deploy? So torch deploy is our answer to a question that we were asking, which is that hey, it turns out that in a lot of cases, we don't really care that the Python interpreter is slow.
Yes, the Python interpreter is slow.
but maybe it's a very experimental model or it doesn't matter that much.
And we just we just wanna, like, be able to run it in a multi threaded environment.
That is to say the only sin c python committed with this particular python program is just that there is a global interpreter lock means you can't run it in a multi threaded fashion.
Besides that, Python is fast enough.
And this is this is often true in a number of cases.
And I'll link to an analysis which was done on a number of models showing that, hey, you know, Python doesn't actually really matter as far as performance goes.
So if you want to run a bunch of models in the same process and being in the same process is pretty important because It just simplifies management of memory, and, you know, you can make sure things get shared in an easy way.
You don't have to go to shared memory across processes.
So single process, but you want multiple Python threads running in parallel inside this process.
How can you do it? Well, torch deploy is the answer to this question.
The use case of torch deploy is pretty niche, and we haven't really tested it that hard in production cases, but it is being tested in CI and Pytorch.
And so if you're dealing with code that interfaces between the boundary between c plus plus and Python, namely c plus plus code that ordinarily doesn't call into Python, but, you know, does you want it to call into Python? For example, dispatched to Python, a project that I've been working on recently, then you're probably gonna run afoul torch deploy as an torch deploy is gonna have to make you think about how to structure your code correctly.
Fortunately, it's not too hard, so I wanna tell you a little bit about how torch deploy is implemented, and then some of the consequences for when you're designing stuff in PyTorch that might interact with Torch deploy.
Okay.
So what is Torch deploy? So Torch deploy is a way to run multiple Python interpreters in your process without them sharing any state so that you can run them each with separate gills.
And technically, Python three point nine's subinterpreters are also an attempt at doing this sort of thing.
But subinterpreters are trying to work with a single copy of the Python in your address space and it's sort of not complete.
Like, they haven't actually gotten it so that each of the subinterpreters is has it got its own state so that you don't have to do the same skill to protect everything.
So Torched Deploy sort of takes a really heavy hammer at the problem and it says, okay.
Well, it's too hard to refactor c Python so that the, like, interpreter specific state is separate and I can, you know, create as many copies of it.
So I'm gonna just gonna take the whole honking python process in its entirety and stamp out multiple copies of it in my process.
ordinarily, you can't do this because, you know, Python is gonna be some shared library.
And if you load a shared library multiple times, well, the normal thing to happen is only load it once.
Right? The whole point of a shared library is a shared library.
You only load it once.
It, like, shows up in one place, and it provides symbols for all the things that you know, it defines as being exported.
So what do you do with torch deploy? What we do is we build a special version of Python that's got all of its stuff bundled up, so all the modules and all the Python code that you need to actually run Python.
But most importantly, it's built hiding all of the symbols.
So you don't actually export any symbols directly from it.
there's just gonna be like a single fixed entry point that we're gonna access with DLSM when we deal open this library.
So we have this, like, blob of code representing a Python shared library that has doesn't export any symbols And what we can do now is we can whenever we need a new copy of the Python interpreter, write it out to a new dynamic library file because, you know, remember if it's the same dynamic library, then the dynamic linker the system dynamic linker is gonna deduplicate all of them.
So write it to a fresh library name and then deal open it without resolving any of the symbols and then manually use Dealsym to pull out the one or two symbols that you actually care about for actually doing access into the interpreter.
And so all of this is mediated by a interpreter class that sort of represents the, like, small set of things you can do to actually run code in your specific Python interpreter.
And the most important thing that it lets you do is it lets you take i values Pytra's internal representation of, you know, like, boxed values that take any sort of shape or size unless you feed it into the pie interpreter so that they turn into pie objects inside.
So what does this picture look like? So when you load up towards deploy and you have multiple Python interpreters going, each of them has a corresponding dynamic library that is their own copy of the Python and because it's their own copy of Python, nothing is shared at all, and so they can all have separate skills.
It's not just, by the way, the c Python library that's in there, you also need Pytorch's Python binding code because the binding code links directly against c Python's API And so, like, because we're hiding all the symbols that can't live in our library itself.
So those also get compiled into this binary and we end up with multiple copies of most of the code in torch slash c circ when you're using torch deploy.
So this is an important segue into some of the limitations and consequences of torch deploy being set up this way.
when you're trying to write code in PyTorch.
So one really important thing is because we're loading multiple copies of the Pytorch library Python, the Python part of the Pytorch library, when we have multiple torch deploy interpreters, It's important that these don't access any shared state and that shared state actually can't deal with multiple copies of the library hanging around.
this is important because we don't actually wanna have multiple copies of a ten, the Tensor Library, or any of the, like, pure c plus plus code.
that c plus plus code we want to have shared across all of the interpreters.
And in particular, for example, if you have code inside the Python library that for example registers an operator to the dispatcher.
That's a no go under torch deploy.
Because remember you would have multiple copies of the torch deploy library.
Each of those libraries when you load them are gonna run their static initializers.
and each of them are gonna attempt to register whatever operator it is you are trying to define inside them.
And the dispatcher doesn't like that.
Right? It only wants an operator to be registered exactly once.
There's also another problem that shows up when you're in the situation like this, which is Let's say you're in some c plus plus code.
It doesn't really have anything to do with Python, and you need to somehow get to Python.
Like for example, you've got a c plus plus struct that was defined inside Pytorch proper, but it has a possibility to contain a reference to a python object that might be associated with one of these these interpreters and say you need to deallocate that pie object when this happens.
Well, if there isn't a dynamic dispatch to the correct interpreter, you aren't even going to know which interpreter you should actually do the pi dot graph on.
Right? Because each interpreter has its own state.
Each interpreter might even have its own representation of the pi object in question.
So you need to make sure you can figure out which one you can actually get.
And so in a previous podcast, I talked about biologic preservation, and I mentioned how there was this thing that we needed to do, which is that when we flip the ownership so that Tensor's own pie objects, we needed to be able to deallocate the pie objects when the C plus plus tensor died.
And so to figure out which interpreter we the pie object is associated with, we have to make an assumption.
And the assumption we're gonna make is that For all tensors in Pytorch, there is going to be exactly one torch deploy interpreter that actually has a pie object representation for this.
This isn't always used to be the case.
In a previous implementation, we actually had it so that every pie pie interpreter could have its own pie object so it was a one to many relationship.
And that was just kind of a disaster because you have to, like, go and deallocate each of the pi objects corresponding to c plus plus tensor if they happen to be owned and you have to take out the gill locks for each of them in turn and there's just lots and lots of opportunity for deadlock in this situation.
But if you can assume that any given tensor only belongs to a single biologic interpreter Well, one, you can still store the pie object on the tensor itself because it's guaranteed to be unique.
And two is, well, because there's one interpreter, you can also like have the chance to remember what the interpreter that it's corresponding to is, and then you can always use that to like do virtual calls into to figure to do things that require the Python API in that situation.
So I've been using this multiple times for different things.
So when we did py object preservation, we used the pi interpreter object, which we're storing on Tensers, which points us to the correct interpreter for torch deploy.
what we are using that for is using that to deck rough the pie object when it goes when the c plus l sensor goes dead.
but in a more recent piece of work dispatched to Python, we're using the Pi interpreter to figure out how to call into the Python interpreter so that we can actually take a call to a c plus plus operator and turn it into a call back into the Python interpreters.
So what's the idea? It's the idea is that we have this dispatcher hierarchy.
It's got all this c plus plus code.
And maybe at the very bottom, you want to override the behavior of an operator and call back into Python.
Well, how do you know which Python interpreted a call with the torch deploy good thing the tensors know what the interpreters they're corresponding to are.
So you just look for a single sensor object that's got a pie interpreter and then use that to do the virtual call into the correct interpreter.
There's a pretty important corollary to this, which is that once you associate a tensor with an interpreter, it is always associated with that interpreter.
Even if the interpreter goes away, like because we decided to unload it, that sensor is permanently associated with that interpreter.
And that makes it easy to make the interpreter recording the red safe because there's a hazard.
The hazard is We have multiple threads and they're all trying to, like, basically allocate a pie object for a tensor at the same time, there's no intrinsic synchronization to this.
And the fact that only one of them can win and once they win that's permanent means that you can just do a plain old compare and swap.
and force the other threads to fail if they lose the race.
Accesses, of course, can just be under a weaker acquire memory mode because once you get a non no result, it is guaranteed that is that result is always going to be that in the future.
One last complication when doing these sort of virtual disectrics.
Unlike traditional c plus plus code where you sort of load up all your libraries, stuff happens, and then shut down kind of happens at the very end, and it isn't really that important.
And It doesn't really matter if you clean up after yourself in the situation because the process is gonna die very soon.
Torsten point interpreters can be spun up and spun down.
And when they're spun down, you will unload the dynamic library that's associated with them.
And that's important because if you have any like spare references to functions from that dynamic library, well, all those functions are gonna become invalid once the library gets unloaded.
And so this is so we don't actually use virtual methods to implement the pi interpreter object.
We use a homegrown v table like implementation with an extra feature that lets you disarm the function pointers when the library unload happens.
So normally, you've got a bunch of function pointers.
They all look great.
And when you unload the process in question, we replace all of the function pointers with no op function pointers that live in the base library so that if anyone else tries to interact with the Python interpreter after it's died, we don't just, you know, seg fall.
we can do a no operation in some cases when it's benign or raise a good error in the situation.
So a lot of tricky stuff going on here, but torch deploy is a pretty cool bullet in our toolkit for letting multi threaded Python processing happen in a single process.
That's everything I wanted to say today.
Talk to you next time.
.
