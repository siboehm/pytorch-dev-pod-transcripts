---
layout: post
title: "Torch Function"
date: 2021-05-31
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Torch Function

Hello, everyone, and welcome to the Pytogitiv podcast.
Today, I want to talk about torch function, a magic method that you can put on any class that makes it possible to override the behavior of everything that happens in the Pytruge library.
Torch function was developed in collaboration with many, many people over many years.
For example, at the very beginning, Dylan Bismalco made a request for sub class preservation in PyTorch, and he wrote an implementation for it.
We didn't end up using this implementation, but the prototype was enough to convince us that we should fund this project.
and a number of folks at Kwan site, namely Ralph Gamer's Pursunnon, and most importantly Hamira Bassi actually took it and, you know, did a implementation that, you know, we actually landed in Pytorch based off of the numpy actually implementation called array function.
So torch function operates very similarly to array function.
So if you know how that works in numpy, you know how it works in Pytorch.
So why would you wanna use torch function? Well, let's imagine that you are writing some code and you want to reuse the functionality in PyTorch.
So for example, we've got all these functions.
Right? We've got Torch dot add, touch dot f f t, tons and tons of API surface like that.
And you might have code that, you know, you've had written against the PyTorch API and you like it, but you wanna do a little bit more.
Right? Like, so the normal tensor behavior works okay, but you wanna extend a little.
Maybe you wanna like keep track of some extra metadata on your tensors or maybe you want to, you know, run some extra code like logging every time you do an operation.
And so you just like to, you know, sub class tensor and, you know, customize the behavior a little bit.
And at the same time, still be able to, you know, run all the good old fashioned operations on PyTorch.
Or maybe you want to completely override the meaning of Tensor do everything on your own, but still be able to use all of the pre existing API that PyTorch provides.
And a good example of when you might wanna do that is say tracing.
Well, you kind of run into trouble if you want to do this.
So if you are just thinking about object oriented programs in Python, ordinarily, you can just change what the methods on an object are, and because Python is duck typed, things will mostly work out.
Right? So, like, if you if I had a tensor, and it supports an add method.
I can just write another object that has a different add method.
And then, you know, if I call add on the object in question, I will just go to my whatever my other implementation is in my new version of the object.
But if I have a function and I pass in an object to that function, ordinarily, I can't overload the behavior of the function in that way because well it's a function and, you know, we don't actually do a dynamic dispatch in that situation.
we always call a single implementation in that case.
And sure, maybe the function might call a method underneath, but maybe not.
Right? Like, and a lot of functions in fighters don't.
They go straight into the c plus plus findings.
And so, you know, there's no opportunity for overriding behavior in Python.
So that's what torch function is for.
Torch function is a magic method that lets you override what the meaning of functions in the torch name space, do no matter what, you know, the object and question is.
So all you do is you write a class, you put a magic method called torch function on it, and then whenever you call a function in torch, Instead of doing the normal behavior, it'll bounce to your torch function implementation and'll and then you can override the behavior however you like.
And in fact, it does more than that.
You know, originally, the only thing we wanted to do was make it possible to override functions in this way.
but it also turned out that it was really helpful to have a generic protocol for, you know, overriding the behavior of all operations on tensors.
not just functions.
Like, sort of analogous to, you know, like, if you wanna do logging, you wanna write some code that works polymorphically over every function and method on Tensor.
You don't wanna have to just write a single you don't have to, you know, do an overwrite for every single method and function you wanna do.
So what Tori function actually does today is it lets you overwrite all method behavior, all function behavior and, you know, write your own custom functionality and then, like, you know, have your code that's written against the PyTorch APIs actually use it in the situation.
Storage function is pretty useful and it's already been used in a number of different situations.
The original request that let us to implementing torch function was someone was writing some code using tensors, and they had sort of units of measure associated with the tensor So the tenses represented physical quantities, and they wanted to, like, you know, classify tenses based on, you know, what was what.
And they had problem, which is that whenever they did an operation on, say, a voltage, like, say, you had a tensor representing voltages and they added two voltage tensors together, even if they were sub class in the beginning when they added those two tenses together, while the sub class wouldn't be reserved.
So originally, the, like, the pitch for this was hey, we wanna be able to sub class sensor and we want the sub classing to be preserved whenever we do operations on classes because that's pretty useful.
Right? like logging sort of works the same way.
Right? If you have a tensor and it's a logging tensor with extra metadata on it, Well, you need to, you know, get it back another logging tensor after you run an operation on it.
Otherwise, your logging will just stop.
So in fact, Tensors have a default touch function implementation that says whenever you have a call onto a tensor that is a sub class of tensor.
We will automatically preserve the sub class for it if all the arguments are that sub class.
Otherwise, we'll just say it's not implemented, and you'll have to figure something out in that situation.
Another situation that torch function has been used for is this tracing use case.
Actually, it's called torch dot FX.
So what is torch dot f x? torch f dot f x is a manipulation toolkit for Pyturg programs.
What it does is it says, okay, you write your Pyturg program using Python.
You can use torch dot FX to trace it into some representation.
You can do some transformations on it, and then you can reinterpret it, re re recompile it.
back into Python code that, you know, you might send a torch grip or something like that.
Right? So it's a lightweight, easy to prototype mechanism, that, you know, lets you do all the syntax manipulation in Python.
And how is torched dot FX implemented? Well, it's also implemented using torched function.
So what Torseshutter FX does is it has a tracer class, the tracer class implements torque function, and instead of, you know, doing all of an normal operations when you call into torch functions or methods, what the tracer object does instead is it just writes down what happened and then gives you a new object that is just, you know, another tracer.
And then, you know, you keep track of things this way.
And then, you know, once you have one of these traces, you can do whatever you want to it.
But the point is you didn't have to modify your PyTorch program at all to run it under torch dot f x, you can still call regular torch functions on the tracer object, and it all works.
Okay.
So that's you know, some of the use cases behind George's function.
How does it actually work? And why is it actually so effective? So let's first talk about how it works because it the the the inner workings of choice option explained a little bit why it's so effective.
So the way touch function is implemented is it's a purely python binding concept.
What do I mean by that? Well, thank you.
Remember, in the very first episode of this podcast series, I talked about how Pytorch Python bindings work.
And so in general, You know, we have this interface where a lot of code is written in Python and eventually you cross over into c plus plus, we translate all the Python arguments into c plus plus arguments and we pass them on below.
So, you know, between there, there's like another level of interaction until you get to the dispatcher, another topic that we've talked about.
in a different version of the podcast.
And so what happens is that torch function is implemented directly on the Python binding layer.
So all of this extra business that, you know, gets you to the dispatcher or the dispatch keys or any of the various subsystems in question, George function bypasses all of that.
Right? Like, it happens exactly when you have the python binding layer.
There's a very pragmatic reason this is the case.
And that's because when we wanna call in storage function, well, storage function is an honest to goodness Python function.
Right? So We need to pass on all the arguments that we were given, and so we need to actually, like, keep the Python representation around.
So if you go any lower, you know, past the Python binding layer, you've lost all the Python objects.
Right? You just have c plus plus objects And then you'd have to, like, reconstruct them into Python objects, and that's annoying.
So it happens at the Python binding level.
But there's a second implication to this as well.
which is that we can actually also override the behavior functions in Python itself.
So what happens is we have a number of functions which are implemented in Python.
So they're not so so the way we implemented George function was we wrote some code generated code to insert into all the Python binding sites that basically said, hey, if you see an argument that doesn't look like a normal tensor, it like looks like some object with a tensor torch function, go call that.
Well, we have a version of that that lives in Python.
So whenever you have a code in Python that's written directly in Python, can write a little preamble at the top that says, well, if any of my tensor like arguments contain something that looks like it has a torch function, then call the torch function instead of the regular function.
And so this way, not only can we bind at the Python binding layer, which is sometimes kinda low level.
Right? Like, You know, we don't the Python mining level is not public per se.
Right? Many of the functions that you see there are, in fact, public PI because they coincide, but many functions are not.
They're just like sort of internal things the way that we get into the c plus plus binding.
Well, you can also override the higher level Python operation that actually explains what's the stuff you actually wanna do in this situation.
And this fact about the torque function implementation that it operates at the Python level and it can operate both at the, you know, level of of Python bindings, but also any higher level abstractions you've written in Python.
It's actually one of the reasons why torch function is so powerful and so popular for doing applications like tracing.
And that's because it preserves the high level semantic structure of your program.
We actually, you know, one of the questions that I often get about torch shot effects is, you know, hey, torch shot effects is just tracing, but don't we already have a tracer in PyTorch.
And, indeed, that's true.
We have what's called the autograph tracer.
This tracer lives in the c plus plus level.
It lives in the dis dispatcher and it also does sort of the same thing as FX which is that it traces things.
So why then is there like another tracer That's FX.
That's built on this Georgefunction thing.
And the answer is FX gets to trace at a much higher level than the autograph tracer because it gets to interpose on actual Python functions.
In fact, you know, one of the things that FX is all about is it's all about tracing n n dot modules.
and because it lives entirely in the Python world, you know, it can actually, you know, record directly what the n n dot module you were operating on was.
when this sort of thing happens.
Right? That would be totally impossible to do in c plus plus because c plus plus has no conception of an n n module.
Right? Everything has been translated into just plain old function calls at that point in time.
Another implication of this is that because it happens at the Python binding layer, you have an opportunity to actually, you know, look at the Python call stack or, like, you know, override the meaning of things that are not even tensors.
So for example, when you call sizes on one of these FX tracer objects, we don't have to give you an integer.
In c plus plus, we would have to give you an integer because, like, c plus plus, if you say you return an integer, it has to be an integer.
But in python, everything's duck types.
So we can actually just return you another tracer object and like do the right thing when it shows up in a trace.
which brings me to my second reason why George's function has been so popular.
And that is because it is in Python.
It turns out that people really, really like to write code in Python.
This was actually it's a little surprising that I didn't learn this lesson given like Pytorch's entire Shtick is that like, hey, you just write normal Python and your programs work, but hear me out here.
So, you know, we knew that Pytorch, you know, from a machine learning practitioner's perspective, you know, it was really useful to write things in Python like that was a essential part of the DNA of Pytorch.
But when we were, like, writing the first version of the compiler, we were, like, oh, no.
Python doesn't have strong static types and we're in the business of writing compiler and, you know, we don't wanna write a compiler without having static types compilos are complicated, you really want as much help as you can get enforcing all the invariance that you have.
So, you know, we decided, okay, we have to write the compiler and c plus plus.
I don't think this is the wrong decision.
Like, you know, having the compiler and c plus plus is really useful.
But what we underestimated was the appetite for, you know, like, sort of short, easy transformations that people might wanna do, you know, like, like, you know, democratizing compiler.
Right? If, like, if you had to, like, learn about type systems and programming language theory and you know, lower level intermediate representations just to like make a little manipulation to your code.
You know, that's gonna keep a lot of people out of doing compilerory things when actually, you know, that's how they should be solving their problems.
And so it turned out that, like, giving these tools to people and letting them do them in part well, so one is a lot of people needed to do stuff like this.
And previously, the only way they could do it was by writing c plus plus.
and that was terrible.
And the second thing is that things were simple enough that, like, doing everything in Python was actually tractable and, you know, people could keep track of everything that was going on.
So, like, hey, you know, like, if you can prototype your entire thing in Python without having to recompile Pyturgy, recompiling PyTorch.
Hey, that's a huge win.
And so that's one of the reasons why people like this a lot.
And like torch function, it being a python level extension mechanism means you don't have to actually, you know, talk to us, Pytorch core, or have to rebuild Pytorch to play around with it.
You can just write your Python function and your research code, write like, you know, just a stock dependency on PyTorch no friendly business going on with c plus plus extensions, and you can do whatever you want, like sort of crazy interesting stuff.
and that's pretty powerful.
That being said, there are some downsides to being a purely python level mechanism.
And the biggest downside and one that we've been working on recently is that you can't take advantage of any of those machinery that lives below the Python binding layer.
And the most important piece of machinery here is autograd.
So, hey, if you override things with George function, you don't get autograd anymore.
Like, if you want Autograph, you're gonna have to figure out how to do it yourself.
That being said, we are trying to figure out how to solve this problem.
And the way we are thinking about how to solve this problem is a concept called dispatch to Python.
The way dispatch to Python works is that you know, we still have this torch function binding layer that works in Python, but you can choose to go into the c plus plus layer.
And in the c plus plus layer, there's a lot of things we can't preserve the python, you know, status of.
Like, you know, if you have an integer argument that's gonna turn into c plus plus integer, Sorry.
We're just gonna completely forget about the original Python object in that case.
But for tensors, we do record what the pie object for the tensors are.
So all we need to do is make sure that we preserve the idea that, oh, this is a tensor that has some extra python behavior on it.
We blast it through our c plus plus dispatcher layers doing autograph, doing batching, everything like that.
And when we eventually get to the final implementation instead of dispatching to our CPU or CUDA implementation, we just dispatch back to Python, translate all the arguments back into the Python and columns there.
And that way, you can actually also take advantage of Autograph while still prototyping everything in Python.
We're still in the early days of working on this.
vunk torch, which is being worked on by Harris and Richard, is a sort of experimental, you know, repository working off of this to give functional transformations to Pytorch.
It's pretty cool.
But, you know, like, I'm hoping that this can be another really cool tool complementary to torch function to let people further extend the behavior of Pytorch on the inside.
That's everything I wanted to say for today.
Talk to you next time.
.
