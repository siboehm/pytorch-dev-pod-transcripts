---
layout: post
title: "Why Is Autograd So Complicated"
date: 2021-06-01
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Why Is Autograd So Complicated

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk a little bit about the constraints slash motivations slash things we are trying to do in the autograd engine in Pytorch.
The autograd engine in Pytorch is the part of Pytorch which is responsible for implementing automatic differentiation.
This is very important for a deep learning library.
You know, if you think about the tagline of PyTorch in the past, it's, you know, a numeric computing library with GPU acceleration with automatic differentiation.
Right? The automatic differentiation is how you can write models in PyTorch run them and then differentiate through them finding out what the gradients are so you can go that way when you're optimizing your models.
It was very, very important and it was built very early in the history of Pytorch and, you know, it is still something that people use all the time today.
Unfortunately, the autograd engine is also very, very complex, and that makes it difficult for people to understand how it works.
And it has a lot of features and a lot of, you know, sort of peculiarities.
And this makes it also difficult to understand.
so difficult that I don't think I could actually, like, technically, explain what is going on with the auto grid engine in just a podcast.
I I'd have to actually write a blog post about it.
I've been promising to write a blog post about it for a while ever since my internal's talk, but it's just It's just a really, really complicated subject.
So today, what I'm gonna try to do is do something a little simpler, which is I'm just gonna talk about a bunch of the things a bunch of the important properties that we wanted out of the autogard engine and some of the implications of those properties.
For example, One thing that, you know, we needed for our autograd engine was it for it to be fast.
Like, you know, we had a version of the Autocat engine that was written in Python and it was pretty slow and we weren't saturating GPUs when we wanted to run networks on it, and that prompted us to port it all to c plus plus.
And so, you know, the Autograph engine lives in c plus plus, and it uses multi threading simply because, you know, at the time it was designed, we needed it to be fast enough to saturate GPUs uncommon, you know, power distributed sorry.
Data parallel training regimes.
So, you know, that was the only way we could get there.
Another thing that the Autograd engine needed to be was It needed a very concise way of writing derivatives for operations.
As I've mentioned before in many other episodes of this podcast, Pietrich has a lot of operators.
And, you know, one of the things that, you know, we sort of insure is the case for every operator someone adds is that it actually has a derivative definition.
And so if you had to write, you know, like, multiple pages of boilerplate just to add a new operator because that was how derivatives were gonna be generated, you'd be in big trouble because, like, we just have way too much code in Pytronch for anyone to maintain in a reasonable way.
And so to get around this problem, we actually built a cogeneration system for autograde engine.
This cogeneration system existed from the very beginning of the c plus plus implementation for Autograph.
And one of the, like, sort of, very famous and, you know, you will probably touch it if you ever add a new operator to Pytorch files in our code base is the so called derivatives dot yamal, which is this yamal file, which for every operator we know how to do derivatives of you write down what the derivative of any given operation is with respect to each of the inputs in the function and question.
And so most derivatives can be written in a single line, and this just makes it really easy to, like, you know, write new derivatives when they're mathematically obvious.
A topic that I should talk about sometime is about the cogeneration pipeline in Pytorch.
And one of the reasons why we have a cogeneration pipeline, which is, you know, not the easiest thing to to understand.
Any sort of meta programming at this scale is not so easy.
But in the case of Auto Grad and I think in the case of most of the uses of cogeneration in Pytorch, it is well worth it.
because without it, c plus plus just doesn't have strong enough meta programming mechanisms, we would have had to have written a lot of code to just implement one of these things.
Like, If we think about, like, when you write something in derivatives dot Yamal, what's going on here? Well, there's a lot of things going on.
For example, when you write one of these derivatives, you can refer to inputs that were given to you inside the that you can refer to inputs that were given to you as inputs to the forward implementation.
What does that actually mean? Well, what that means is that when we're running the forward of a model in PyTorch, when you refer to an input in the backwards formula, that means we have to save that input so that it's still available when you actually, you know, refer to it.
in the backwards pass.
So, you know, we have to save it.
We have to, like, write a struct.
We have to put a place where we can save the thing.
We need to actually save it in the forwards thing.
We need to get it out again.
and plug it into your formula.
So that's a lot of moving parts and the cogeneration handles that all for you.
So you could just, you know, it looks like you're just closing over closing over, you know, the input at that that time.
Like, you know, one way to think about derivatives is they're, like, just higher order functions.
But, you know, in c plus plus, that's not so easy to do.
So we have a lot of things to make this simpler.
Another thing that Pytrux needed to support when doing automatic differentiation was views and mutation.
Right? So like one of the really big things part of Pytruda's DNA is that you can take out views from Tensors, so these views you know, don't allocate new data.
They share storage with the original tensor in question, and you can also mutate them.
So, you know, like, if you wanna fill in just a single row on a tester, you could view out that row and then just run fill on it.
And our automatic differentiation system actually needed to work correctly even when people are doing views and mutation.
There's a few ways, senses in which I mean it needs to work correctly in a situation.
One sense it needs to work correctly in the situation is just sort of basic correctness, which is just to say that, you know, you have tensor that you wanna save for backwards so that you can use it later.
And then if someone goes ahead and you know, scribbles all over it with garbage sometime later in the forward pass.
Well, you're just gonna get garbage out in the backwards pass.
If you try to reuse that buffer exactly as is.
And no, we don't wanna copy out variables when we save them because that would be expensive and remember we want automatic differentiation to be fast.
We don't wanna, like, impose, you know, that kind of overhead on users.
And also, you'd probably run out of memory if we were doing that.
So to make sure this doesn't happen, we have this mechanism called view counters, sorry, version counters, which record, you know, what how many mutations have happened to a tendering question, so that when we save it, we can say, oh, you know, three mutations have happened.
And then when we come back, we check you know, is it still only three mutations? If it's five mutations, that means someone's mutated it in the meantime, and we can give a good error message in that situation.
But there's another more important thing that we need to do to support views and mutation with automatic differentiation, which is that we can actually support differentiating through mutations in some situations.
For example, if I have a tensor and I, you know, take out a view and then write out that view with that tensor which requires gradients.
The result is that my, you know, base tensor which I wrote into, now also requires gradients.
Right? Because if I use it as part of my loss computation, that bit of the tensor that I wrote in using that view now contains data that, you know, tracks its providence back to that tensor that I originally requires grad from.
And so there's actually a pretty complicated apparatus in autograd.
We're making sure we can keep track of what automatic differentiation happens in the situation when you do a mutation on a view with something that requires grad.
And this is if you remember the podcast about inference mode, this was some extra metadata that you actually don't need in inference mode, and inference mode lets you dispense with doing that.
But, you know, when you're doing normal automatic differentiation, you need this information and so we track it so that you can, you know, do all the things you expect to be able to do in Python.
There's some other performance stuff that we do to sort of make reverse mode automatic distribution work in a predictable way because At the end of the day, whatever reverse eighty engine is is it's this multi threaded c plus plus, you know, opaque engine that, like, runs your code and you don't really know, like, what is going on with it because it's not written in Python.
You can't debug it.
And furthermore, there's no, like, direct sequence of calls you make.
Right? You just call in the backward, and then a whole lot of stuff happens in that time.
So one of the things is it needs to be possible to debug problems in your autograph in a reasonable way.
Right? Because Yes.
We say Pytorch is this eager mode framework and, you know, like, you can just write code and write debug statements, but that doesn't really hold true when you do reverse mode a d because all this stuff is happening without any corresponding source code.
By the way, tangent, a research project at Google, for doing source to source, automatic differentiation, one of their pitches is like, oh, you know, we'll take your Python program and turn it into a differentiated Python program that you can just debug directly if you need to debug problem.
So Pyturbine doesn't do that.
So what do we do instead? Well, we have a bunch of extra mechanisms built into AD such as anomaly mode, which anomaly mode normally you use to debug why are NANS showing up in your tensors But another thing that it does is it, you know, keeps track of what backward operations correspond to what forward operations.
So when something fails in a backwards operation, I'll tell you.
And by the way, this was the forward operation, the back trace that actually caused that situation.
Another thing that we do is we have a pretty sophisticated hooks mechanism whereby you can insert arbitrary pieces of Python code at any point when you're running your backwards, you know, computation and say, hey, you know, give me what the gradient is at this point in time.
and let me take a look at it, you know, maybe modify it if I'm doing some weird gradient scaling or something like that.
But really, you know, I can just take a look at it and figure out if, you know, it's what I expect or not.
a way of inserting, say, debug print statements.
And so, you know, these things are not conceptually complicated but a lot of effort is spent inside the autograd engine.
So if you're like reading the code and you're like, oh, what is all this Hooks business and this anomaly mode business? Well, it's not important to the core algorithm, but it is important to making sure users get a good experience when using the autograd engine.
There's also some really unusual features that are autograd engine supports, which also add to the complexity of the implementation.
So one of these things is so called reentrant execution.
What does reentrant mean? Reentrant means you're inside some sort of procedure and you wanna call back into the previous user again while you're inside.
And so you're reentering while you're already inside.
So reentrant execution in the context of automatic differentiation, the autograde engine is you're in the autograde engine.
You're executing, you know, your backwards functions one by one.
And then inside one of those backward functions, you actually execute autograd again.
Why would anyone wanna do that? Well, one one answer to that is, you know, like, Autograph is just this operation.
Right? Like, it computes the derivatives of a function.
And so, like, that just is a normal mathematical computation that, you know, you should be able to do anywhere.
In in the other in other words, Grad should be composable.
But there's another, like, sense in which reentrant execution is really useful, and that's for checkpointing in Pytorch.
Checkpointing is this trick for reducing the memory usage of your models that says, hey, I'm not gonna record the saved variables Remember that? Right? I'm not gonna record the same variables for everything in my network.
Instead, I'm gonna force the network to recompute the variables when I actually get to them I'm training away compute so that I can reduce the amount of memory I use.
So how do we implement re entrant x how do we implement checkpointing in PyTorch we do it with ranching execution.
What we do is we run our Pyderidge program, we run the forwards, and we just don't save anything.
And then when we come back in the backwards and we need to figure out how to, you know, execute the backwards formula, while we've fail to save anything.
So what we do is we just rerun the forwards again and then reengineally call backwards on it.
to get the actual backwards computation computed in this case.
This was implemented by Priya Goil back in the day, and people use it.
And so it's, you know, one of the most important use cases for reentrant execution in PyTorch.
there there's a bunch of, like, complicated stuff where you can actually get into these this bad behavior where you keep reentering over and over again, and then you blow your stack space.
And there's also some logic in the Autograd engine to deal with that.
One last thing that the Autograd engine supports.
which is that, normally, Autograph is this thing you think of as running on a single process on a single machine.
Right? Like, you just run Autograph.
You've got your entire graph.
Well, in the distributed setting, we actually have an implementation of distributed autograd, which allows you to distribute autograd across multiple processes across multiple nodes in case, you know, your program in question is too big to run on a single processor.
And so there's a sort of like specialized version of Autograph called Distributed Autograph, which uses many of the same implementations, but override some important stuff that makes it possible to just run autograd in this distributed fashion.
So that's pretty cool, also complicated in its own right.
You can read more about it if you're interested.
So why is Autograph so complicated? Well, one is that there are a lot of features, there's a lot of performance requirements, And, you know, when you put it all together, there's just, you know, you have to work pretty hard to do something like this.
So that's one of the reasons why, for example, in my previous podcast, it was really, you know, interesting for people to be able to reuse our autograd engine directly because, hey, you know, we've already done all this stuff.
So you'd like to reuse it in that situation.
But, you know, there's also like something to be said about a simple implementation of autograd that, you know, is hackable.
Maybe doesn't have all the efficiency.
Doesn't have all the features.
But, you know, it just has the core algorithms for Autograph.
That's a good idea too.
And we have a bug report that's tracking this issue.
So hopefully, you've come away from this with a little more appreciation of you know, why autograd is more complicated.
And so if you're ever looking at this code and you're like, oh, what is this business with hooks? What is this business? with, you know, this view metadata.
What is this business with this multi threading nonsense? Well, hopefully, this podcast has giving you some clues about why those things might actually be there.
That's everything I wanted to say today.
Talk to you next time.
.
