---
layout: post
title: "Torch Vs Aten Apis"
date: 2022-04-04
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Torch Vs Aten Apis

Hello, everyone, and welcome to the Pytorch Step podcast.
Today, I want to talk about the a ten and the torch APIs in the PyTorch library and how they affect how we think about API design as well as our intermediate representations that we send to graph mode compilers.
Now, you may not realize it, but Pytorch actually has two APIs.
The first API is the API that I'm going to call the torch API and it's the one you know and love.
It's the Python API that, you know, you use when you interact with our library as a normal Python developer user.
So it's a documented Python API that uses all of the regular idioms that you'd expect from Python.
And in fact, we have a limited amount of programmability for this API as well via torch function, which if you don't know what that is, you should go listen to my podcast about TorGEfunction.
But basically, we can override the meanings of these Python API functions, including functions that are entire nearly written in Python.
That is to say.
They they're just playing Python and they have a little check at the front that says, you know, if the If any of the inputs are tensor sub classes, then defer to them to figure out how to implement this function.
All of these things constitute what we call the Pytorch torch API.
and it's you can also get a list of all of these methods via the torch overwrites module, which gives you a bunch of overwritable functions and methods that you can actually change the behavior of when you sub class a tensor.
So this is the normal API that everyone knows and loves and you might be thinking, If I want to work on Pytorch internals, then clearly, I'm going to expect to see a lot of functions that have that reflect the torch API.
Well, that's not quite right.
When you're working inside PyTorch's internals, you're more likely to work with a different API that I'm going to call the a ten API.
a ten in this case stands for the a tensor library, which is a internal c plus plus library that, you know, sort of Pytorch's Python front and is built on top of.
So the a ten API is a more limited API in a sense that instead of being the entirety of the Python language on anything that is supported in Python is supported in the API, the agent API operates on a restricted set of types called the Jitschema.
This restricted set of types originated from the fact that we were working on torch script compiler front end, and we didn't wanna support every single type in Python.
So the Jit schema says, what types that the JIT API supports.
But these types map both to Python as well as to c plus plus, and they're selected to be some limited subset that's tractable for us to map to all of these languages.
So from the start, the functions that make up the eight in API has its limited set So you won't see a function, for example, like a map in the eighteen MPI map.
In the PyTorch API, it's a very obscure function, but it takes a function a Python callable and runs it on every element in your tensor slowly, but, you know, it's something you sometimes wanna do.
We can't do that in the eight ten API because we don't have a concept of a function that's portable across languages.
So there's similar limitations like this.
So the agent API has limited type system, and in fact, basically, every function that you can think can think of in the Pyturg API maps to one or more operations in the a ten API.
Sometimes this mapping is quite obscure.
For example, prior to Joel Schlosser refactoring our convolution implementation, we had maybe thirty different internal eight ten convolution operations.
Whereas, you know, in the public fighter j p i, there was, you know, one or three depending on if you count count one d, two d, and three d as being separate things.
So the eight in API is is exhaustively enumerated inside native functions dot yamal, and we it's not documented.
Like, if you squint, most of these will be similar to the PyTorch Python API.
but some of them will be different and you'll sort of have to read the code to find out what the difference is.
But the difference then is that because the eight ten API is what we actually operate on in C plus plus.
Most of our internal subsystems, for example, autograph, are written in terms of the eight ten API.
So for example, if you wanted to look up a derivative formula for something in PyTorch, you wouldn't find a derivative formula for a function directly in the torch API, the Python API.
Instead, you would have to find what a ten function it mapped to and then look up the derivative formula for that a ten function.
Hopefully, pretty obvious.
Most of the time, sometimes not so obvious.
Now although we said that the Python API is overwritable via torch function, the eight ten API is also overwritable by tensor sub classes, but you use a different API for doing this namely torch dispatch.
And torch dispatch sort of also interposes at this lower level where all of the subsystems are already finished running.
So it's more appropriate for that situation when, you know, you want Peter to have done most of the work and now you just wanna do a little bit of extra information in this case.
Although the eight ten API is primarily oriented at, you know, existing at the c plus plus level and being the, you know, library implementation that the Pytorch Python API is implemented on top of, we also do expose agent operations directly via the torch dot ops.
module.
The torch dot ops module essentially has a sub module for every name space of operators, and the eight ten operators are put in the eight ten name space.
So for example, if you wanted to call the native ad, you would say torch dot ops dot a ten dot add.
And that would go through a different code path than the traditional fighter ship You usually don't wanna use this API directly.
It's mostly intended for people who are one programming torch dispatch, where when you get called in torch dispatcher, given one of these torch dot ops dot a ten functions to tell you, hey, you know, this is not a regular Python torch API, this is an eight ten API, or perhaps when you use our custom operator registration API, the torch set ups gives automatic Python bindings, whereas most of the Python bindings in the traditional Python API are automatically generated.
So if you think about PyTorch as just an eager library, it's not too hard to understand torch versus a ten.
So torch is the front end.
the Python API and internally get back ends to a ten, which is a lower level c plus plus API It's a little more factored, but it might have some more internal functions for various things we need to do.
And depending on what level of interposition you want, in PyTorch's internals, you might use torch or you might use a ten.
But there's another way to think about these APIs and they are That way is to think of them as intermediate representation dialects.
When we have Pytorch, eager mode matters a lot, but graph mode also matters.
And Pytorch also allows people to target, you know, take PyTorch programs, turn them into graphs of operations, and then send them to various back ends.
And now because we have these two APIs, you also have two ways you can end up with your IR.
You can end up with an IR that has the torch API, or you can end up with an IR that targets the a ten API.
And depending on your trace acquisition mechanism, you'll get one or the other.
So how do you end up with the torch API, AKA, the python, the public API? Well, if you use the torch FX, tracer.
That tracer operates at the Python level.
It actually doesn't even go and execute any of the internal operations.
and it will collect up a FX graph that contains all references to public torch API functions.
This is by the way one of the reasons why FX is such a popular graph representation for Pytorch.
It's because, you know, when you look at these graphs, they look exactly like what you'd expect to see, you know, based on what you know about Pytorch's Python front end.
However, there is a downside to this.
because FX tracing operates purely at the python level without interacting with any of Pytorch's internal subsystems, there's some basic functionality that you don't get when you're working with the FX tracer.
For example, if you want to take a graph and look at the backwards for it, there's no easy way to do this with the basic FX tracer.
And so there's another trick tracer, called the AOT autograph tracer, which can take a FX graph and retrace it through the c plus plus implementation using torch dispatch to get out a backwards graph.
But this backwards graph won't be for the Pytorch Python API.
It will instead be for the eight ten API.
So you'll get it actually also uses FX.
So FX IR can be thought of as a container format.
which can have several dialects in it.
And so in in this case, when you use AOT Autograph, you get out a FX graph that contains eight ten operations.
More concretely, when you look at the various, you know, function calls in the graph, instead of being calls to torch dot ad and torch dot sub, they'll be calls to torch dot ops dot a ten dot ad and torched dot ops dot a ten dot sub.
Actually, technically, you'll also even know which overload you had.
This a ten ops IR, FXIR, is closely maps to torch grip IR, which also operates on the level of ATN operations.
And then depending on your back end, you will have some back ends that expect FXIR in the torch form and some back ends that expect FXIR in the eight hand form.
For example, if you have a FX graph mode pass, for example, like the quantization pass, that's gonna expect code that is in the targeting the torch IR.
But if you have, for example, some pass that was previously targeted at torch grip, for example, NVFuser, it'll be easier to get there using the a ten i r.
And so when we have a these two i r dialects, we can start think about, you know, can we transition from one to the other? Now, clearly, we can go from torch to a ten because that's basically the process that happens when we execute our eager code, we take a bunch of user calls to the torch API and then, you know, do some infrastructure to get it down into lower level A10 calls.
And we can trace through those using any trace mechanism that operates at the A10 operator level, whether it's torch dispatch tracing or, you know, lazy tensor tracing for example.
But what about going from a ten to torch? Well, hypothetically, this should be possible because the a ten API is a well defined API and the torch API is a well defined API.
So you should be able to implement the agent API in terms of the torch API.
Unfortunately, no one has actually gone around and done this, but we think it would be a useful capability and we want to add it to Pytorch at some point in the near future.
Another consideration is how dynamic or static the IR produced by the various tracing mechanisms are.
When you do FX tracing, the graphs you get are very, very symbolic.
For example, when you call dot size on a FX proxy, you don't get back an actual two pull of numbers.
You get a symbolic proxy object that represents the sizes, but in fact, it's just gonna record your subsequent uses of the sizes.
And the fact that everything in FX tracing is symbolic is one of the reasons why sometimes you can easily trace models because while they're relying on actually knowing something concrete and FX isn't willing to give that information to you.
In contrast, essentially, all APIs that go through PyTorch's internal subsystems and c plus plus all require very concrete values for all the sizes, strides, d types that are involved.
That's because in our c plus plus implementation, we literally have, you know, lists of in sixty fours floating around and how are you gonna replace that with some sort of proxy object? work by Nick Caravallo is working on extending our in home representation to allow for symbolic integers so that we can trace some level of dynamic shapes.
This work is in early stages, but we're hoping to get it done this year.
There's one more teaser that I wanna leave you with, which is that we are looking at adding a third API.
So you might be thinking, wow.
Why do you want so many APIs? So one reason is that the agent API, despite, you know, being more lower level than the torch API is still essentially intended to be basically the same thing as the torch API.
So for example, torch dot ops dot a ten dot add, that's still a broadcasting type promoting operation.
And for some backends, that's still a bit too implicit.
You might want to have your type of motion and broadcasting be explicit so that the back end can easily say, oh, I see this is a non broadcasting ad or oh, I see this is a broadcasting ad.
So the prem ops API is a concept where we have a even smaller even more simple layer of operations under a ten.
Now, obviously, decomposing operations like add into their constituent type conversions and broadcast is not good for eager remote performance.
So the prim ops formulation is not intended for regular usage in PyTorch, but instead for use with compilers, which can recover performance even if you've atomized a, you know, point wise operation into a lot of any pretty small parts.
And also for symbolic analysis applications, where you'd like to, you know, target a simpler set of operations that is more that's more factored and easier to understand and then have it sort of take your complicated surface pyrites program and de sugar it into a bunch of small operations that are individually easy to analyze.
So that's a lot of the stuff that's going on right now, and that's everything I wanted to tell you about today.
Talk to you next time.
.
