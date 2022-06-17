---
layout: post
title: "Automatic Mixed Precision"
date: 2021-05-18
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Automatic Mixed Precision

Hello, everyone, and welcome to today's edition of the Pytorch dev podcast.
Today, I want to talk about how automatic mix precision is implemented in Pytorch.
on the request of one of our listeners.
Thank you very much.
So what is automatic mix precision? AMP or automatic exposition or internally referred to as auto casting is a feature by which when you write your models in PyTorch, we will automatically downcast some of your parameters to lower precisions so that your models can run faster.
So what do I mean by that? So imagine that you've got a bunch parameters.
Right? Your parameters are probably floating point numbers, which is the normal thing to do in this situation.
And you wanna, like, you know, do a matrix multiply with the parameters and your input ordinarily, you would just do, say, a float float matrix multiply and, you know, that would go however so much fast.
But, you know, Nvidia being the tricky people they are, they actually have a faster implementation of matrix multiply.
That happens if you give it a half precision input and a floating point precision input.
Half being, you know, a representation of floating point numbers that uses only half the number of bits.
And, you know, because there's less bits, there's less compute to do.
And so if you actually have silicon for it, which NVIDIA GPUs do, it can run faster.
So if you pass it in, in this half precision way, your stuff magically gets faster.
So that's mixed precision operations.
But the automatic part of automatic mixed precision is you actually don't have to do anything to your models to get the benefits.
automatic mix positions API is this context manager.
You say, okay, turn on AMP, and then magically your modules use mixed precision when it's appropriate.
What exactly does AMP do? Well, the heuristic that's applied here is actually pretty simple.
Basically, AMP says, okay.
When it comes to operations involving parameters, this is the situation where the x true resolution on the parameters tends to not be so useful.
Right? Like we use floating point thirty two bit floating point numbers to represent parameters because we need to be able to do updates on them.
But as far as the computation for the neural network is concerned, most of that precision is not actually used.
And so it turns out, and, you know, this is not obvious.
You had to run experiments and show, actually, this is profitable.
It turns out that you can just task your floating point parameters into half precision, run your network this way, and it will use less memory, it will run faster, and it will train just about as well.
So Michael Carrillo and co at NVIDIA actually did an implementation of AMP as part of their Apex toolkit.
you know, advanced Pytorch extensions.
And at some point, you know, Emily was, like, talking to me at the Pytorch Devcon And he was like, hey, you know, I wanna put this in core, like, how can we do it? And at the time, we had been working on this new dispatcher thing.
Yes.
I I I talk about the dispatcher a lot because my team composability works on dispatcher features.
Like, that's kinda what one of the big things we do.
So I was like, oh, you know, there's this interesting new thing called the dispatcher, and I think it gives you enough rope to actually implement automatic mix precision.
and, you know, we went back and forth a bunch of times with a different few different proposals.
But in the end, we have this implementation of AMP It works transparently.
It has the same API that Apex had, namely, a context manager.
You don't have to know anything about it when you're writing operators.
It's a complete extension on top of operator writing.
So, like, if you're just a plain old fashioned operator, then some normal behavior will happen in that situation.
Like, you know, you don't have to worry about it.
And and that's that's important too.
Right? Because not all algorithms have faster mixed precision implementations, like matrix multiply and convolutions, those actually have tensor core algorithms, and they can go faster and have precision.
But a lot of things don't.
And so, you know, there's no need to deal with them in that case.
And then furthermore, it's extensible in the sense that if you have external libraries like, say, torch vision, which doesn't live in Pytorch, they can also be extended to use AMP.
And it's it's all extensible.
Right? Like, sort of AMP is this, like, capability layered on top of PyTorch.
operators are extra pieces of functionality that are layered on top of Pytorch as well.
And the dispatcher lets us, you know, put the square together, we don't have the expression problem.
We we can actually do the extension in both ways and then fill in the last corner of the square.
Okay.
So how does it work? Well, let's remember what AMP wants to do.
Right? So what AMP wants to do is when you turn on this mode, when you turn on this context manager, we need to change the behavior of all our operations that know about, you know, AMP.
And this will be a fixed set of operations that, you know, curistically, we know are useful to do AMP things on.
And we need to change the behavior to instead of taking parameters directly.
We say, okay.
Well, I don't wanna take this parameter directly.
I wanna cast it.
to a half precision and then run the operation on it.
So algorithmically, that's what we wanna do.
Like, sometimes when I get an operator, I want to cast things and then, you know, use the cast.
And furthermore, like, you know, if this parameter is being used a bunch, I wanna cast the cast.
in this situation.
So I'm not repeatedly converting it unnecessarily.
So how do we go about doing this? So step one is how to actually intercept operations when you want to, you know, when a context manager is being set.
And this is actually, like, the textbook use case of what we call mode dispatch keys.
So what is a mode dispatch key? A mode dispatch key is a dispatch key that typically isn't put directly on a tensor itself, but instead is something that gets put into our thread local state that, you know, basically, in the dispatcher, we have a third local say that lets us include dispatch keys and lets us exclude dispatch keys globally, no matter what the tensor inputs are.
So to, you know, Enable this context manager, the AMP context manager when you turn it on, says, okay, put the autocast key into the local TLS that says, okay, whenever I do operations, I wanna include the TLS.
And then if, you know, the Autocheski is not in the local TLS.
Well, then I just bypass all these kernels.
The second recipe here that we need to know about is what are we gonna do about all the operators that, you know, don't know anything about AMP? In this situation, we want to just sort of fall through to the default behavior.
We just wanna run the normal operation in this situation.
So there's another tool in our toolkit in dispatcher, and this is called a fall through kernel.
So fall through kernels are a kernels that we put in the dispatcher that say, hey, don't do anything here and said just fall through to the next valid implementation for the dispatch key in question.
And, you know, why is there a next valid implementation? Well, all the dispatch keys are ordered in a sense.
Right? So there's a priority.
You do auto grad first, then you do the CPU key.
And in this ordering, autocast needs to live somewhere And so, you know, when auto cast, you know, when when we when we have a kernel and then we hit auto cast because hey, you know, AutoCast mode is on.
If that kernel doesn't do anything special for AutoCast, fall through just says, okay, go to the next key in that case.
And most typical auto cast kernels are gonna go ahead and do some operations, and then also do a re dispatch.
They're gonna say, okay.
Forget about doing any more AutoCast stuff.
I'm done with AutoCast.
Go ahead and do whatever the next thing you're gonna do was.
Cool.
and actually fall through is implemented very efficiently because the way we determine what what dispatch key to, you know, call into in the dispatcher is we actually look at a bit set of all the dispatch keys, and we just do a find the first set bit So when you have a fall through installed for a kernel, we actually just don't set the bit inside this bit field, and you don't actually have to, you know, go ahead and do the dispatch and then realize oh, there's nothing to do here, fall through the next one.
It's completely free, so you can always add these fall through keys without paying any cost.
Okay.
So we've got a way to intercept all operations when a mode is set using the TLS key.
and we have a way of making sure operators don't actually call the A and P kernel if we don't know anything about them.
Namely, we have a fall through key and we've already started at this fallback.
Right? So any anything that, you know, doesn't explicitly have an autocast key just shows the fallback.
what about the actual implementations of operators that do have fallback keys? Well, it's not too hard.
Right? So intuitively, you know, we've gotten all our inputs and we need to decide, you know, whether or not we're gonna cast some of them to have precision.
And then, eventually, we need to call into the actual operation that is underlying the auto cast implementation.
So what are the steps of this? Well, you know, and naively, the the the main the meat of the algorithm, right, is, like, looking in an input and deciding if you're going to do it to have precision.
Unfortunately, there's no like cut and dry rule for how to actually decide if half precision is gonna be useful or not.
We have a few rules of thumb in the a dispatcher tutorial, like, you know, matrix multiplies and convolutions are likely to be profitable with half precision.
If you're doing reductions, you probably want them at higher precision because, you know, catastrophic cancellation is more of a problem.
But, you know, really, really it's, you know, testing things out and seeing what works well on actual models that you wanna run things on.
Okay.
So let's say that you decided that, okay, this parameter should get casted to have precision.
if it is a parameter.
So we have a helper function that attempts the cache cast.
And what it does is it says, okay, you know, is this a parameter namely, you know, is it a least salarable? Make sure it's not a view.
We actually forgot to put the view check-in, and this really resulted in some hilarious bugs where people were taking views of parameters in loops and we were continually adding things to the cache.
parameters are good because there's a fixed number of them.
You don't have to worry about there being too many of them.
and they stay live for the entirety of the computation.
So they're it's usually safe to cash them because the lifetimes line up.
Okay.
So you look and see if it's a leaf, if it's not a view, and then all you need to do is go ahead and cash, cast it, and, you know, put it in a cache.
And the cache is just a good old fashioned hash map, and it gets cleared at the end of every training loop, namely when you, you know, exit auto casting.
And that that's pretty convenient.
Right? Because at the end of the training loop is when your parameters are likely to update and therefore when all of the cast entries are likely to be invalid.
Okay.
So how is that actually implemented in PyTorch? Well, there's a bunch of operators that, you know, do need auto casting support.
And actually, you know, the co union write in this case is very regular.
And so at the time that Emily was working on auto casting, we still had a lot of bugs in our boxed fallback.
The the mechanism I caught talked about in the previous podcast, which we used to implement conjugate views.
So that didn't sort of work out.
And it was okay because there's only a fixed number of operators that they really needed.
So instead he just wrote a little template Right? So he he has this template meta program that takes in the name of the operator, takes in what the type signature of the operator is, and then, you know, constructs a new wrapper function that, you know, does the operations based on some policy.
Right? Because some functions we wanna cast to have precision, some functions we wanna stay as float.
some functions, you know, if there's a explicit d type, we wanna use it.
This is just a template that picks apart the arguments, you know, looks through them, checks for parameters, cast them to have precision.
then sets a dispatch key guard that says, okay, don't ever go to AutoCast anymore, and then red dispatches.
By the way, on the re dispatch, typically, the re dispatch is going to auto grad.
And the reason we want re dispatch to go to auto grad is because auto grad is gonna save some inputs for backwards.
And we would much appreciate it if it saved the half precision inputs because that's half the memory you're spending, saving things for backward.
Okay.
So, you know, we've got our dispatcher which lets us, you know, set up this autocast key.
That's a mode that only gets you know, turned on when we need them.
We talked about what to do about operators that don't need auto casting, and we talked about what to do about operators that need auto casting.
And actually, that's it.
Like, AutoCast is a really, really short implementation.
There is not very much at all to it at all.
And you know, it's a single file in our code base called autocast dot c b p.
You can read through it.
It's got all the interesting details.
Really, the hardest thing is just, you know, figuring out what the policy you should apply on the operations should be.
And shortly after we added AutoCast to Pytrich Core, you know, Franchesca Masa, for example, pull gave support for AMP and torch vision.
So it's it's actually fairly well supported even throughout the library ecosystem.
AMP was so influential that actually Intel is working on a CPU version of AMP.
not for half precision because there isn't really good silicon for doing half precision on CPUs, but b float sixteen does pretty good on CPUs, especially when you're vectorizing.
So they want a version of automatic mix precision that does b flip sixteen on CPUs.
And they're just, you know, modifying the the existing CUDA autocasting code to work in this case.
So that's how autocasting works.
Take your parameters, cast them to have precision.
cash that cast and then, you know, use it throughout.
And once again, the way that it is integrated into PyTorch in an orthogonal way is by using the dispatcher, which lets us, you know, layer on extra pieces of functionality that you don't have to care about unless you know, you you actually do wanna care about it and then you can write implementations for it.
That's all I wanted to say for today.
Talk to you next time.
.
