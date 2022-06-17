---
layout: post
title: "The Life And Death Of Variable"
date: 2021-05-12
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# The Life And Death Of Variable

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I wanna talk about a topic that I've received two requests for when soliciting topics to talk about in the podcast.
And this topic is variable.
Actually, it's a kind of strange topic to be talking about because if you look at Pytorch today in the Python front end, there actually is no variable anymore and that's because we got rid of it.
It was a banner feature in Pytorch zero point four And then a bit later, we actually got rid of it in the c plus plus code.
Although, there's still a bunch of places where, you know, we still talk about variable.
That's just because we've been too lazy to rename all the type names and the code base.
But it's still really useful to know the history behind, you know, variable because there are a lot of, like, strange APIs that still exist because of the fact that tensors were structured in a different way.
And it's also kind of informative just to like look at how the format of tensors has evolved over time and also where they might be going in the future because I would definitely not be the person who would say that we are in a perfect state.
So where does our story begin? Our story begins a long, long time ago even before the existence of c plus plus autograph in Pytorch.
So in Louis Torch, Tensor was represented as a c struct.
And remember this thing, right, how the t h library in Louis Torch has a bunch of c code that's munched about with a preprocessor.
Well, that's true for the data type as well.
So when we wrote t h code, we had a c struck and there was a separate c struck for every d type we supported.
So there was a t h float tensor, a t h double tensor, a t h end tensor, and so forth and so forth.
This made life really hard if you wanted to write code polymorphically over different sensor types, but it didn't matter because we were just rewriting all of our code every time, when you, like, wrote code in t h, we just, you know, redefined the macros and then stamped out different versions of the code.
So along comes Pytorch and we're still using the good old fashioned t h tensors.
And Zach comes along right and he wants to build a ten c plus plus library.
And one of the things that he needs in the c plus plus library is he wants to be able to write code polymorphically over device types without templating them.
Because you see in c plus plus, if you write templated code, you don't actually get to type check the contents of your template.
Right? Like, the way c plus plus works well, until, you know, c plus plus twenty whatever concepts come along, the way c plus plus works is that when you write a templated function, Seapeless Plus only checks the stuff that isn't related to the template.
Anything related to the template is deferred until you actually instantiate a template in question.
So, you know, c plus plus templates are famously a source of really bad error messages.
And so, you know, we had a bunch of people we're previously writing all of our operations in Python, and we were gonna try to write them in c plus plus.
And so, like, forcing them to template all their code on d type would have been a really really bad idea.
So, like, if there was one good idea in the a ten library, it was this, don't parameterize your tensor type on d type.
Okay.
So we had a single tensor type and we put it all together and we said, okay, there's gonna be a single tensor impulse that represents all the d types in question and that's gonna be pretty cool.
But remember that the t h library and Zach's A10 library didn't know anything about automatic differentiation.
And at the time, AD was implemented entirely in Python, so there was like no concept of this in c plus plus.
And this was true in lower towards as well.
AD was a thing that was implemented in lua, not inside the libraries themselves.
calls.
And so when Sam came along and he was like, oh my god, you know, autograd is too slow.
We need to make it faster, and we're gonna do it by porting into c plus plus.
He was in the position of needing to write an implementation of autograd in c plus plus rather than in Python.
And so the most obvious way to do this was to preserve the abstraction barrier that was enforced upon us when autograd was within Python, namely that the tensor subsystem knows nothing about automatic differentiation.
So let's think about it.
Right? Like, say you have some library that gives you a tensor object and lets you do various basic operations on them.
Well, what if you want to augment this with some notion of history and a notion of an autograd tape that you record graph operations to do later when you wanna autograph on them.
Well, if you have this strong obstruction barrier between the tensor and the AD system, you can actually modify your tensors to, like, add the new metadata you need.
So what are you gonna do? You're going to wrap them in a variable.
So variables where just this wrapper around tensors that, you know, gave all the extra metadata that you needed to get yourself working in the situation.
And so it started off as a requirement, right, because autogard was written in Python.
And then when we moved everything to Salesforce, well, the most easy thing to do was to preserve this obstruction error.
So, you know, We had everything in c plus plus, but, you know, it was still, like, implemented as there is a variable wrapper and it is on top of the a ten library.
Right? They even lived in separate dynamic libraries if you remember the dynamic library podcast.
So, okay.
we've got this variable concept and, you know, it's like zero point three in Pytorch days.
And, you know, we've got tons of people using Pytorch and they love it.
and we keep getting all these questions about when should I wrap my tensers and variables? What's the difference between a variable and a tensor When do I use dot data? Could they get a tense route? And what we discovered is that it was actually really really confusing for people to have to manage both variables and tensors.
Now, it is really like easy way to organize the code when we were implementing it.
But the problem from the user experience expect perspective is there's too much expressivity.
There's too much freedom in this representation.
Namely, you can have a tensor.
You can have a variable that doesn't require a grad.
And you can have another variable that does require grad.
And the problem is that, you know, each of these three states, the tensor state and the varial doesn't require a grad state, these states are basically the same.
Like, semantically, they do exactly the same thing.
The only problem is, while, you know, while you've got this variable thing, you got this tensor thing.
So people have to, you know, worry about, you know, switching between these two modes.
Even though, like, you know, if if they're just thinking about, like, what is it they wanna do? Right? Like, what they really wanna do is they want some tensers to record gradient gradients and some to not.
And, you know, having to deal with this extra distinction that doesn't do anything useful, that's pretty confusing and they don't like that.
So we were like, okay, in zero dot four, we wanna get rid of variable.
Right? And we wanna just make it so that when you're writing byte rich code, you don't have to deal with, you know, remembering if you've wrapped something in a variable or not.
So we got rid of variable.
How did we do it? Well, we cheat it.
The way we cheat it was we just said, okay.
Well, we got this big c plus plus implementation with variable to tensor and, like, you know, it's a ton of code to refactor.
We don't really feel like refactoring it.
Also, we didn't actually know how to do this refactor.
here's what we're gonna do.
In Pytorch, we're only going to provide you variables.
So like this thing that we call tensor, secretly, it's a variable.
And, you know, that means that, you know, we've eliminated this illegal state when you don't actually get to you know, look at the the illegal state is now a bear tester.
Right? Because all you have are variables or variables with the cards grad.
and that worked pretty well for a while.
So we had this problem though, which is that, like, in the Python API, there's only tensor But if you, like, dive down to c plus plus and you're like a c plus plus writer, there's actually still this variable concept.
And so one of the things that, like, we really wanted to do was, you know, hey, like, maybe we want the Python and the c plus APIs to look the same.
Like, maybe that's a good idea.
and we can do it.
But there's a problem.
And here's the problem.
The problem is that the way we implemented Autograph is via this unwrapping operation on variables.
So the idea is that, like, you have a bunch of variables floating around.
You do some operation on them.
And when you do the operation, well, you know, you've got a variable.
So you go over to the variable implementation.
And let's say you're doing the implementation of add.
So we we're gonna set up some autograd graph, right, to, like, you know, record.
And then we wanna actually run the original the original code that actually implements the add kernel.
So how do we do that? Well, inside every variable, it's there's a tensor that you can unwrap from it.
So we just unwrap the tensors from the variables, and then we call add on those.
And those are just tensors, they're not variables, and so we can actually get to the actual kernel in question.
So how do we do this for if there's no separation between variables and tenses, if every tenses are variable, how do we actually do this? And you think to yourself, oh, yeah.
You know, Ed, what you should just do in this situation is you should, like, make a super call.
Right? Like, you you've got your autograph code and then you just wanna call super cool and cool and add and that'll bounce you over to, you know, whatever the you know, on the the parent implementation is ostensibly doing the actual edition.
But we have a lot of operators in PyTorch, and many of these operators actually call other operators in their implementations.
And when they call those other operators, you don't actually want them to hit autograph in the situation.
You want them to go and you want them to go and go straight to the, you know, non autograd actual kernel computation.
Right? Because it's sort of like, you know, once you do an autograd call, you've actually you're done.
There there there's no, like, internal autograd bookkeeping you need to do.
it's a single atomic unit in this situation.
So you wanna bypass everything underneath.
Those of you who have read my dispatcher talk know how we solve this problem.
So Wolfgang implemented the c plus plus tensor variable merge.
And the way we solve the problem was we introduced some thread local state.
So what we said was, okay.
What we're gonna do is we're gonna have these variables and we're gonna, you know, do our autogarty stuff on them.
and then we're gonna set some thread local state that says, don't do any more autograph stuff.
That's actually what auto non variable type mode used to do.
We we've killed that now, so check out the inference mode podcast for more details on that.
So we so we set this some TLS And now whenever we do function calls, we just check is, you know, the autograph skip TLS bit set.
And if it is set, then, you know, we go and go to the actual kernel instead.
The actual implementation is more complicated than that.
But if you're just thinking about Autograph, this is all you need to know.
And so in that way, we didn't actually have to do any unwrapping step to actually, you know, make it so that we stopped running the autograph code.
and started running the Tensor Code.
Now there were a few other complications.
So one of the things that was supported in the variable API is this data attribute.
So what does the data attribute look like? Well, you know, if I have a tensor x, then I can say x dot data and I'll get out, well, Who knows what it does today? But in the old days, right, if you had a variable, well, you know, x was the variable, and then x dot data was the tensor on the inside.
And so if x was a thing that requires gradient.
Well, extra data is a plain old sensor.
Obviously, it doesn't require gradient.
So we had to, like, figure out, like, what exactly these things should do in the New World order because we're not wrapping variables anymore.
So there aren't any there there's no tester inside waiting to, you know, burst out.
Sorry, the tester was not inside you all along.
So what are you gonna do? Right? Well, we just looked at those semantics and we're like, okay.
Well, you know, what is this exat data? Well, it aliases the same storage as the original tensor.
So it's kinda like a alias call, but, you know, it doesn't require gradient even if the variable required gradient.
So it's kinda like a detached call.
So you know, and, you know, what what about the version counters? Well, version counters are a concept on variable originally, and then we put them on tensor.
And so What are version counters? Well, that's a long story for another time.
But if you know what version counters are where we store version counters and variables.
When we put them on tensors, If you took out the data, the the inside sensor of the variable, you would actually disconnect from the original version counter.
So we also simulated that behavior.
basically, we, like, looked and we're, like, what is all the observable behavior you could see when you did a dot data? And then try to figure out what that would look like in a universe where, you know, there are no variables everything's just a tensor.
So that was done and it sort of worked for a while.
We were in this weird another state where we had collapsed the representation.
So there was only one there was only one tensor representation rather than a variable wrapping a tensor we hadn't actually expunged all the variable classes from the code base.
And then later, I actually went and finished off the job and got rid of all those wrappers.
And then that's sort of where we are today.
Right? So we have tensors.
It's a single struct, but this struct has a few fields really one field dedicated for letting you slot in auto grad metadata if you actually want it in the future.
This data is not actually defined in Tensor.
It still lives in a separate dynamic library.
The in the Autograph folder in C bake, and It contains a bunch of extra data.
And so if you don't actually require autograd, we don't bother allocating all this data and you can save a bunch of time.
By the way, one of the reasons why, you know, inference mode and, you know, no ground mode is faster than, you know, if you're recording autograd.
And so that's like basically the state of Tensor today.
So where could we be going in the future with this? Well, one of the things that people have been looking into recently is how to make it so that you can nest automatic differentiation repeatedly in a style that is not the same old style that we normally support double backwards and pie charts, namely, you retain graph and then you back prop through the graph again.
So more like a jacks style, like, you know, repeatedly differentiate a piece of code ahead of time.
So how can we do that? Well, we've got a prototype that knows how to do this.
And actually, it's done by Well, who would guess, wrapping the tensor into multiple levels of gradient tracking to make it work out? So I don't know.
Revenge of the wrappers, I suppose.
So that's all I wanted to say about a variable today.
See you all next time.
.
