---
layout: post
title: "Torch Deterministic"
date: 2021-07-21
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Torch Deterministic

Hello, everyone, and welcome to the Pyturg Step podcast.
Today, I want to talk about the determinism mode in Pyturg, which allows you to run Pyturg's computations in a fully deterministic fashion so that if you run Pytouch programs again, you'll get the same result.
What's the point of determinism? Ask no one whoever had to debug a very hard to reproduce problem because the problem was not deterministic.
Deep learning programs are already very hard to beat debug because well, there's a lot of stuff that is going on that you don't really directly have access to as an engineer.
And so, you know, if you forget to add a constant in some area, you might not find out about this except that your network trains a little bit more slowly.
And so if you're, you know, doing a four hour long training run and, you know, only into hour three, do your gradients start exploding and you get lots of nans everywhere, well, it's going to be very painful to debug this problem in most circumstances.
So determinism makes at least part of the problem easier, which is that if your program is bit for bit deterministic if it always produces exactly the same bits every time around, then, well, okay, maybe, you know, you don't find out about the problem.
until three hours in.
But at the very least, every time you try, you will get the same result.
Because the only thing that's worse than a problem that only happens three hours into your training is a problem that only happens once every ten times three hours into your training.
Ugh.
So torch deterministic was a proposal that was made by Sam Gross a very long time ago, and we didn't really do very much with it.
because there's a very annoying thing you have to do to make this feature actually come into being, which is you actually have to audit all of the operations in byte merge.
and look for ones that are not deterministic and then do something appropriate in the situation.
So lots of kudos to Kurt Moller who actually picked this up and, you know, God saw it all the way through the end.
He's the one who made it all happen.
So what's the basic concept between torch to dot deterministic? Well, there's a few things to talk about what you want out of a way of running your programs deterministic.
First off, you don't want determinism to be on all the time.
Why? Because being deterministic is actually quite expensive.
There are a lot of algorithms where if you allow for a little bit of non determinism, they can run much, much faster.
And when you, you know, make things run determistically, while you're gonna get quite a bit of a slowdown.
And so, you know, there there's actually a very delicate balancing game.
Pritchard does with regards to its defaults.
Do we give people, you know, all the knives and run really fast and make it easy for them to do things that are wrong? or do we actually, you know, try to prevent errors and, you know, try to make sure people, you know, don't do the wrong thing.
And sometimes we trade off performance or, you know, making it harder for people to do the wrong thing.
But determinism is not one of those things.
We do not give you determinism by default, you have to ask for it explicitly.
Another question is, one of the things about nonnaturism in your network is you might not even know about it when you when one of these things happens.
So there's sort of two parts to torch deterministic.
So one is just letting you actually, you know, use deterministic algorithms whenever they are available.
But second is just identifying when non deterministic code is being run, so you can know oh, yeah.
This training run is using that function.
That function is non deterministic.
Maybe it doesn't even have a deterministic implementation, but at least I know about it.
and I can, you know, ask my system error if I use something that doesn't actually have a deterministic implementation.
the framework level implementation of torch dot deterministic is pretty simple.
There is a context manager that, you know, you can use to turn on the, you know, worn or error on determinism.
And then everywhere in our code base where we're about to do a non deterministic operation, there is a line that just says, hey, alert that, you know, this is non deterministic.
And then depending on the setting, if, you know, you're supposed to error or if you're supposed to warn, you'll get one of these other things.
And in some cases, if determinism is requested, we can route to a different algorithm and there's just an if statement that does that.
very, very simple.
So really most of the juice is in deciding one that this was a good idea and two actually following through and editing all the algorithms.
There's actually not that many different types of non alternatives in the library.
So one of the most common ones is the back end libraries that we use especially in CUDA, are often non deterministic in these in some situations.
So the, like, classic example of this is convolution.
Kudianan, you know, has a lot of algorithms, and some of its convolutional algorithms are non deterministic.
And in fact, there's even prior to the, you know, generic determinism flag, there was a Cudian N deterministic flag, which was specifically about you know, foregoing use of the faster algorithms so that, you know, you would use one of the, like, deterministic compilation algorithms.
In other cases, you know, that are not library code, the most common reason for nonlinearism is an atomic addition.
So to explain why atomic additions can cause non determinism, it's important to know a little bit about floating point numbers.
floating point numbers are not commutative.
Repeat after me, floating point numbers are not commutative.
You actually think about you know, what it would take to actually write a floating point implementation.
This kind of makes sense once you think about it because Like, let's imagine that, you know, you have a, you know, very, very small quantity and you keep adding it until eventually it becomes a larger quantity.
Like, zero point one plus zero point one plus zero point one plus zero point one and so forth.
And eventually, you know, you can get something fairly big until, you know, your precision falls off the cliff and you don't have enough precision to know every time when you're adding plus zero point one, you, you know, actually can put it in.
Because this is a limited number of bits.
Right? The whole point of floating point is you can change the amount of precision you have.
So, like, if you're close to zero, you get a lot more precision if a really big number, you have less precision.
But if you're adding, you know, zero point one to, you know, a trillion while you're probably not actually, you know, like, there's just no space to represent it.
in the floating point representation.
Well, if you, you know, do a bunch of these additions and you get to a reasonably large number and then you add it to a large number, you know, you can expect to see the contribution of all those incremental zero point one.
But if you start with the big number and then you keep adding zero point one to it and each time you don't actually pick up the floating point number because well, There's, you know, no change because you don't have precision to represent it.
Well, you know, clearly, you're gonna get a different result in these situations.
By the way, there's some kind of interesting ways to work around problems like this even with limited precision.
One interesting way is to sort of randomize whether or not you take up the result or not depending on if, you know, it's just too small for the precision.
So like say, you're adding one with a million and you don't have enough precision to represent one, but you do have enough precision to represent ten well, maybe you would just increment only a tenth of the time, non deterministic.
Of course, this is really terrible for determinism.
So let's get back on track.
So because floating point edition is not communicative, if you have any operations that are like, hey, run a bunch of stuff in parallel and then automatically accumulate it into some buffer, which, you know, is doing some sort of reduction.
Well, that's going to be non deterministic if you do it the obvious way, which is with an atomic, you know, edition in the situation.
So most cases of use of atomic ad in CUDA, those are non deterministic.
And so it's like it's like actually super super simple half the time to figure out if something is non deterministic.
Does it use atomic ad? Oh, it uses atomic ad? Well, it's probably non deterministic.
And that's really all there is to the feature.
Right? It's, you know, a context manager that sets some global variable that triggers the behavior.
and then a bunch of code everywhere that, you know, says whether or not something is deterministic or not.
One of the things that we do you know, would accept patches for.
And in fact, in the last half, some work that some folks did was for some of our internal training workloads They added support for a lot of deterministic versions of operations that didn't previously exist.
Kudos to them.
Great work is yes.
In some cases, we will just hard error if you ask for which deterministic.
And if you provide a deterministic version of the algorithm that we can use in place of it, even if it's slower, that just, you know, makes the deterministic feature more generally applicable.
I also think of torch dot deterministic as a really good sort of API model for other types of things in PyTorch which we might want to do.
In particular, there is nothing intrinsically wrong with non deterministic operations.
It just happens that sometimes you want to know if you're about to assemble into one of these things.
So it will be nice to be able to just easily set some flag to then be told about all these situations, maybe as warnings to just find out about all the sites or errors if you like absolutely absolutely cannot abide with a non deterministic result.
Well, there's actually a lot of other behaviors in a framework like PyTorch, which have a very similar property.
For example, Natalia again Moshein is working on a version of this but for CUDA synchronizations.
what's a CUDA synchronization? Well, that's when you have some CUDA computation and you're like, hey, GPU, please finish all your compute and then send me the result back to CPU so I can go look at it.
That's, you know, something that will happen implicitly sometimes in your petrives program it can trash your CUDA performance and it would be really nice to know when this has happened.
And it's as simple as just making sure everywhere these synchronizations can happen.
We have a test and so you can set a flag and then, you know, have it raise a warning or raise an error.
So once again, super simple but it goes a long way.
So this is like I don't know.
It's not very glamorous, but I think it has a lot of value to our users.
And so I wanna encourage people to work on this kind of feature because, hey, it really does pay off.
Okay.
That's everything I wanted to say for you today.
Talk to you next time.
.
