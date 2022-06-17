---
layout: post
title: "C Frontend"
date: 2021-06-08
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# C Frontend

Hello, everyone, and welcome to the Pytorch dev podcast.
Today's topic is a listen request, namely a discussion about the trade offs behind the design of the c plus plus front end.
So before we start, I have to first explain what I mean by the c plus plus front end because there are a number of different ways you can interpret this.
In one sense, the c plus plus front end is the tensor class that is inside PyTorch and is used to, you know, actually undergo the implementation of all our kernels and all of the plumbing that is in PyTorch.
So this is Tensor provided by the a ten library originally developed by Zachary Dorito.
And it's a really important piece of what we think of as the Siebel plus front end.
So I'll spend some time talking about the philosophy there.
but there's a second part to the c plus plus front end and this was added after a ten by the by Peter Goldsboro and what it is is basically everything else beyond, you know, just the tensor class.
Because if you think about Pytorch's library, we don't just provide a tensor.
We also provide a module of extraction and an optimizer abstraction that you can use to easily structure your neural networks.
And, you know, people use tensors a lot, but they also use modules a lot.
And so that matters a lot.
when you actually wanna read write real code.
But we wanna start with talking about tensors because that's simpler and it sets the stage for some of the design constraints.
that happened when we were designing the rest of the c plus plus front end.
Okay.
So let's talk about a ten.
So where did a ten come from? So, A10 came from this idea that, hey, we were writing all of our internal code in Pytorch in this very terrible language called t h where we had various macros for your tensor types and it was all done in c and you had to write your code and then compile it multiple times for every d type you wanted it to be supported on and you had a manually rough count and it was all terrible.
And so the model behind A10 was, okay, let's use c plus plus instead of c, and use the abstractions that C plus plus gives you to actually make a nice API for doing manipulations on Tensor.
But it went a bit further.
So There were a number of other TensorLibraries in c plus plus at the time, eigen being one of the most influential ones, and we didn't wanna do that.
we the idea that Zac had was we want to have a tensor type in c plus plus that is just tensor.
It doesn't record any d type information.
It doesn't encode any dimension information.
And the really important thing about doing it this way is now you can write polymorphic code on various d types and various dimension sizes without having to template your code.
Because, well, you know, when you're writing C plus plus, if you have a type and it's got some parameter on it like you're doing a vector and it's got some, you know, type of the elements in it.
If you wanna write a function that is generic on the types, you have to write a template function because c plus plus is gonna instantiate it for every copy of the element type you use.
And it gets worse and worse because the templates don't actually get typed checked.
You have to wait until they actually get instantiated with the type in question related to yeah.
Type check.
So it's just much harder to write code in c plus plus if you are using templates.
that is until c plus plus comp sets come around.
But, you know, we were c plus plus eleven at the time.
So, oh, so much trouble like and one of the things that makes it really hard for newcomers to c plus plus to write c plus plus is the really horrible secure template error messages.
So if we just don't put that information in Tensor, if we type a raise tensor, then people don't have to worry about that.
So that's the, like, first main innovation of a ten, which was don't do templates, just type erase everything, and it's okay.
Things will work out in the end.
Another really important philosophy that went into the design or tensor is we really wanted it to look as much like Python as possible.
Right? So if you, like, wrote some code in Python, like, I have a Tensor dot ad b dot mall c, Right? Like, that's something you could write in Python.
No problem.
We wanted that to be exactly the same way in c plus plus.
So people who came in not knowing very much c plus plus but needing to write their code in c plus plus because remember, this was at the time we were trying to start moving all of our Python code into c plus plus.
So we were in desperate need of c plus plus programmers, but everyone knows how hard it is to actually find grizzled c plus plus veterans that know everything about the ownership model and see the slides.
There's just, like, not that many of them.
So the closer to Python we could make the code, the easier and more accessible it would be people to start riding kernels in c plus plus.
And so one one implication of this is tensor like eighty tensor as seen in PyTorch is not the traditional notion of a c plus plus type, which is a value type, where if I were to, like, do a copy construction on it and actual shall I copy what happened, No.
It's a reference type.
So we actually organize most of the main user visible types in Pytorch into two types, a tensor type, which is the reference type So if you copy it, you just, you know, are copying the pointer, and then Tensor Imple, the Imple type, which actually contains all the metadata in question.
And so you'll see this separation in storage, storage storage, ample, and also in modules, module, ample, module.
So you get reference semantics, Equally works this way you expect it to in Python, and people are pretty happy.
One last thing about the c plus plus API.
which is that we want our calls to look a lot like Python.
And for the most part, function calls are the same.
But one thing that Python has that c plus plus doesn't is keyword argument support.
So we needed some way to actually simulate keyword arguments.
And I'm getting my timeline a little bit mixed up here because we added keyword argument support to the cnosis API after we actually did the initial version of a ten In in particular, the reason why a ten didn't have keyword argument support was it wasn't obvious how to do it.
And the sort of most important structure that guts used everywhere in PyTorch, Tensor Options is designed explicitly to let you do this sort of keyword argument style arguing passing in Python.
How does it work? It's just a struct.
It can be default constructed to have nothing in it.
and then you can set via cellular methods various attributes on it.
So, like, dot Tensor Options dot d type, blah, dot device blah will set up things so that you actually get a sensory options with that d type and device set, but maybe not other the other keyword arguments.
And we actually design Tensor Options to be a value class, so you don't have to worry about like mutation or someone mutating it under you.
It always functionally returns you a new tensor options.
It's only two words large, so it's not a big deal to keep creating new copies of this tensor options.
Okay.
So I've established the basic ground rules that, you know, the A10 library wanted.
Right? Which is that no templates Don't don't do templates, so it means we need a Type A Race Tensor and make the Tensor API look as much like Python as possible.
We actually even wrote a manifesto like this about this, writing c plus plus, writing Python and c plus plus.
So with these two constraints in mind, let's fast forward a little bit in time when Peter Goldsboro was working on the C plus plus front end proper, namely module support.
So at the time, there was a project going on at Facebook research.
The Starcraft project -- Mhmm.
-- they were doing reinforcement learning for Starcraft.
And they had a problem, which is that, you know, what they needed to do was they they needed they had a simulator for Starcraft.
An actual game instance of Starcraft, actually.
And they needed to feed it information from the reinforcement learning model that they were training at the time.
And they needed this to go as fast as possible because, you know, like the faster you can do the simulator, the the faster you can actually do training.
And so CPU overhead really mattered here and parallelism in multi threading really mattered here because running lots of simulators.
And this was just completely impossible to do in an efficient way in Python.
And so they actually started writing a little layer on top of the a ten library, which, remember, we call only had tensors and that that that's it.
All it is is a tensor library.
I'm called AutographPP, to make it possible to do automatic differentiation on these things and to, you know, actually structured modules.
And so at the time, Peter Galsborough was like, you know, hey, c plus plus random is a really good idea.
And there are a lot of people who might be interested beyond the Starcraft project.
And we took the, you know, learnings from their version of the c plus plus front end and built it into the c plus plus front end that actually you can use today as part of Fire Church proper.
So we ran into a few questions when we were trying to figure out how exactly modules should work in c plus plus.
Like, there are a number of problems.
For one, we already have modules in Python.
If we want modules in c plus plus Does that mean the Python module should call into the c plus plus modules? Well, maybe that's not such a good idea because a lot of people take modules in Pytorch.
They copy paste them into the research code and they hack on it.
This hackability is really good when you're writing Python.
And if we actually moved all the implementations into c plus plus, then, you know, well, people can't just copy paste things.
Right? They didn't actually have to compile some c plus plus or like look up an old version of high touch where there was still the Python implementation.
So we decided we didn't want to get rid of the existing Python modules because hackability was really important there.
Another question was could we write a transpiler to take these Python modules and transpile them into equivalency plus those modules.
And that just seemed like too much complexity for things to be worth.
So we decided, okay.
We're just gonna re implement all the modules that are in Python and c plus plus.
For better or for worse, because now you've got two versions of the code, you gotta update both of them in this situation.
We have another problem when you're trying to implement modules in c plus plus, which is that, you know, Python has all of this meta programming stuff.
If you recall my previous podcast, unforged dot n n, I was like, hey, you know, what does module do? Well, it tracks parameters.
And really, the, like, most important thing it does is track parameters so that you can collect them all up and pass them to your optimizer.
But the way Python does that is by overriding the meaning of setting attributes on the module, so that it can, like, then, you know, sideband, like, recorded in some field that says what all the parameters of modular.
Well, how the heck are you gonna do that in c plus plus? The answer is you can't.
So need to adjust the API a bit.
So the way the c plus plus runner works, right, is it asks you to register parameter when you register a parameter and that just sets up the extra metadata tracking necessary to tell what the parameters in question are.
Another problem, which is similar to the quarries problem from the Tensor case, is that modules also often have a lot of arguments that you want to, like, express, like, keyword arguments.
And unlike factory functions, which tensor options is sort of oriented towards, which have a fixed set of keyword arguments.
If you create everywhere, every module is a little different.
So there's a bit of work in the c plus plus API to make it easy to define, you know, options, objects that you can, you know, use setters to set in what the options should be, and then eventually passes to the module in question to make things work up.
And one last thing.
Right? Modules, we argued a lot about whether or not they should have reference or value semantics.
in the end.
Right? Python and c plus plus.
Right? Like, these Python modules should look the same as the c plus plus version.
So All modules also are split in the module module impulse split.
And that's why there's a macro that you need to call to actually, you know, bring the module into question.
So what is what what what's the up upshot? Well, we started off riding the Siebel's front end for Tensor, and we had some design principles, namely right python and c plus plus.
And we extended it to modules in c plus plus perhaps a little imperfectly because modules are a lot more complicated, but we were still trying to consistently apply this idea to the entirety of the C plus plus front end.
And I would say that's sort of like the the main idea.
Right? Like, you're not gonna get exactly the best performance that you could have gotten by writing really idiomatic c plus plus, you're gonna get something pretty good and certainly much better than, like, if you were writing Python and had to, you know, worry about the guilt.
and that's good enough for a lot of researchers.
That being said, there are some performance challenges to writing code in this way.
And actually, Scott Walchuck, a engineer over in core Infra who has been on loan to us on the fighter project has been working on reducing overhead in our framework.
And some of the stuff that raises a lot of overhead is related to writing Python like eighty plus.
So let's just check out a few of these.
So one problem that we have is that rough counting is really slow.
Why is Ref counting really slow? Well, Python Ref counting is actually really fast, but there's a trick behind it, which is that because there's a global interpreter lock, Python Ref counts are non atomic because you can just assume that they're gonna be protected by this lock.
In Siebel's plus, ref counts are typically atomic because you want your ref count of objects to work across multiple threads.
So, you know, you actually implement the ref counts as atomic things.
And incrementing and decrementing atomic fields, that is expensive because you have to tell the processor to actually send the cache line back to the main memory in question.
Oh my god.
So, like, that's that's a huge hit.
So, excess ref counts are a problem.
And one of the difficulties about writing code in the Python style where you only have the Tensor concept, which is a pass by reference type, a shared ownership type, is that, well, a lot of the times people are just going to start, you know, doing ref count bumps willy nilly because that's kinda what you did in Python where it was cheap.
Well, it's not so cheap in c plus plus.
And we've actually developed a really interesting way around this problem.
So conventionally, the way you would have solved this problem in c plus plus is that you would have, you know, made a strong distinction between the the actual thing that contains the data and a shared pointer to that data in question.
And then you would force everyone to use the right pointer, whether it's a raw pointer or shared pointer, or unique pointer, or some arena allocated pointer, and force everyone to, like, do all this juggling around.
We came out the problem with we've got this tensor type Everyone expecting is expecting to be able to do const denser and for San.
So we we have to have an actual tensor at the end of the day can we reduce the amount of rough counts going on in this case? And the answer is yes, because we actually implemented rough cutting ourselves using an intrusive pointer class.
And what we can do is we can build wrappers on top of Tensor, for example, maybe owned Tensor, which dispense with the ref counting because the ref counting ends up being, you know, an incorrect or a decorative call.
So you just skip the rough counting when you're in this container type depending on what's going on.
So for example, if I have a maybe owned Tensor, which is actually just a reference to some Tensor, it's non owning, then I have the destructor of Maybury destructor of Maybury own Tensor.
Just leak the Tensor when it gets destructed.
So don't trigger the normal destructor of tensor, which would deckgraft, just skip the deckgraft entirely.
And you can actually build a bunch of other things.
There's actually a PR out for also exclusively own tensor.
Right? So this is kind of like unique pointer, but unlike unique pointer, it's piggy backing off of a shared pointer.
So, you know, when you know you only have that pointer, you don't have to actually engraft and decrepit.
but then you can promote it into a regular shared reference.
That's very much like unique pointer in this case.
But at the end of the day, it's still a tensor And so you can still, you know, forget about all of these pointer distinctions and pass around constant references to Tensor without having to rewrite all your code.
So, yeah, I would say if we were gonna do this project again, we would probably think about not writing all our code in c plus plus and perhaps writing it in some language and then writing a compiler stack to compile down to the actual machine code we want.
and, you know, figuring out how to make it run really fast and because we because compilation time is a huge problem.
You don't actually wanna be, like, spending a lot of time compiling.
But that's a huge infrastructure outlay.
And I don't think there's any way we could have gotten to the point we are today, not using this concept of writing c plus plus in Python.
So I still think it was a really good call.
It saved us a lot of template headaches.
It really made it possible for a lot of people to write code in our framework in c plus plus.
but, you know, like, there's always something better you can do.
That's everything I wanted to say today.
Talk to you next time.
.
