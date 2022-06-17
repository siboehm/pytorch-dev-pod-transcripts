---
layout: post
title: "Multiple Dispatch In Torch Function"
date: 2021-08-09
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Multiple Dispatch In Torch Function

Hello, everyone, and welcome to the Pyturbine Podcast.
Today, I want to talk about multiple dispatch in torch function and how you can use it to make sure your torch function implementations play nicely with others.
So if you don't know what torch function is, I highly recommend go and listen to the torch function podcast that I did a few weeks ago.
The short version is torch function is a way to overload the meaning of torch functions when you make your own custom sub classes in Python.
And so when you're writing a torch function, there is an interesting problem, which is what if you sub class tester one way and you say I want the behavior to be this and someone else subclasses the tester another way and says I want the behavior to be some other thing and I pass both of these tensors to the same operation like say I add a logging tester with a, you know, unit of measure tensor, what is supposed to happen in this situation? If we look at the behavior of Python in situations like this, on normal method overloading, we realized Python is a single dispatch language And so traditionally, there is a distinguished argument, the self argument for which you actually do the implementation on.
So let's say that I have I'm adding two objects together, a plus b.
Well, what will happen is I will call the add magic method on a because on Python oriented towards, you know, preferring the first argument into this situation.
And a is responsible for checking if it actually understands how to deal with the second object in question.
If b is a sub class of a, chances are a is going to just go ahead and treat b as if it were an a without using any the extra behavior from b.
Of course, this can be horribly inflexible sometimes, and so Python added another way to handle situations such as, what if you said one plus some object instead of some object plus one? Well, clearly, you can't override the underscore underscore add on the one literal.
So what Python also has is the right side versions of match magic methods such as r add, which say If the implement method isn't implemented on the first object in question, try it again with the second object in looking for the other implementation or add instead of add.
And so what will happen is when you say one plus some object, First Python will attempt to run the operation using the implementation from one.
One is gonna say, I don't know how to add to the sum object thing.
So I'm just gonna return not implemented.
And then Python will try again with the second argument, calling our ad on that argument, and this time it will work and you'll actually get a successful dispatch in this situation.
So to recap, in stock Python, Most method dispatch is a single dispatch.
And if you have a normal method on a function, that's what's going to happen.
But sometimes there is a need for multiple dispatch and Python has this sort of convention which is, you know, well, try the operation on all of the objects in question.
And, you know, if one of them says, I don't know how to do it, try it on the other one.
So binary ops and, you know, ops with many tensive arguments are galore in, you know, the Porsche library.
Right? Like, whatever we had to deal with addition in Python, while we also can add two tenses together.
And so when torch function was originally designed as array function in the Numpy ecosystem, It was designed with an extra mechanism for making sure multiple dispatch would work in this situation.
Here's how it works.
And remember, it works very similarly how Python simulates multiple dispatch and certain magic methods.
When you call an operator that is torch function overloaded, The first thing we do is we collect up the classes of all the tensor arguments in it because that's all of the possible implementations of torch function that may be used in this situation.
We look and see if any of these classes are sub classes or other classes.
This is important because, well, let's say that I have an a and I have a b that inherits from a and b together.
It's better for me to try the b first rather than the a first because b might have some special handling that overrides the behavior of a stock a operation.
Other than that, I pick some arbitrary order to run the Photonics functions on just subclasses first And then I go ahead and run them one by one, and the first time one doesn't return a not implemented error and actually returns an actual result, that's when I actually return that result for real.
However, torque function implementations can say, I don't know how to deal with this.
and pass on the baton to some other class that might be able to handle it later in the implementation.
Unlike stock Python, we don't have special versions of George function if you are in the first argument or the second argument or third argument, Georgefunction is a class method.
So it can always be called no matter what where the class in question lives in the argument list.
So, you know, as an actual implementer of George function, you're responsible for going over the arguments and making sure if they are actually your object in question or if they're a normal tester or God forbid, there's some other class that you don't know how to deal with.
So let's imagine that I'm writing a logging tensor And a logging sensor is very simple because it just prints something and then just wants to go ahead and run whatever the operation was before.
So a logging tensor is kind of universal.
Right? It works in any situation.
And so we don't need to be very restrictive about what kinds of other sub classes we can deal with.
So a logging sensor might go ahead, look through all the arguments, find the logging sensors that are in them, log what their values are, and then go ahead and unwrap them and call the function again on the same arguments as before.
Remember calling the same function as before, make sure that if there are other sub classes involved, those can get a chance at it.
The logging sensor just removes itself from the picture.
or let's say you're some very special tensor that is implemented like as a back end into some accelerator or some custom back end.
Well, you're probably not going to be able to deal with arbitrary sub pluses.
So what you should do in the TorGE function is when you are processing it, you should go through all the types that were passed in and check that they are all exactly your type or maybe, you know, a tensor type.
if you see anything you don't support, you should return not implemented instead of raising an error or anything like that.
This is not super obvious to do when you're just copy pasting code, but if you keep it in mind, it's actually pretty simple.
It's just a little bit of extra error checking that you need to add to torch function and make it compose well.
with other implementations of George's function.
And of course, it's not a magic bullet.
Right? At the end of the day, someone needs to be able to handle all of the arguments in question.
So if, you know, you have a bunch of extensions and none of them know how to deal with each other, then well, that's fine.
You'll just get an error saying that there wasn't any touch function that actually implemented this.
The key thing about multiple dispatch is that you can retrofit new functionality onto the system that you may not have had before.
So imagine that, you know, someone's gone ahead and built a torch function sub class that does some extra behavior and then you're a further extender and you're like, oh, this is a great idea.
But I if only I had another class that I could customize the behavior even more, Well, that class knows about the first touch function implementation, and it can write generic implementations that work in both cases.
And in this way, you can post facto add more functionality onto the system that, you know, perhaps the original implementer of some class didn't anticipate.
And this is one of the things that people like a lot about multiple dispatch.
It's this ability to solve the expression problem.
I just, you know, putting giving people a place to put the completion of how feature a interacts with feature b.
So, multiple dispatch in this way is kind of cool.
And remember that I said that we we always run sub classes before their parent classes because, you know, they're more specific.
But otherwise, the order of the multiples dispatch is unspecified.
And PyTorch is allowed to pick whatever order it wants.
But in general, most operations you're gonna do on a tensor aren't commutative.
And so it's kind of it's a bit tricky if, you know, you actually are going to run these in any arbitrary order and you still want them to be well specified.
So what really is gonna happen most of the time is most of your operations that, you know, don't know about each other are just gonna say not implemented when they see something they don't support.
And it's only really the things that, you know, know about each other they'll have a very specific ordering in mind.
But there isn't a situation when you do want to be able to make custom sub classes of tensors and you want them to be composable and you want control over the order in which they run.
And this is called Funktorch.
AKA, jack style, composable transformations on functions.
One way to think about what Funktorch does is it creates a bunch of new sub classes like batch tensor and grad tensor, which, you know, imbue the meaning of operations with different things.
Right? Like batch tensor, takes in what used to be a single example series of operator calls and turns them into batch versions.
And a grad tensor takes what used to be a simple, forward only, series of calls and then also computes the backwards at the same time when you execute those calls.
The composition of these passes matter.
It matters if you do a v map and then a grad, which is traditional good old fashioned, you know, training over batch versus a grad and then a v map, which is a more exotic type of training called per sample gradients, we actually compute a gradient for every single sample.
You don't average all together in one big loss.
And the whole pitch about Funktorch is that these transformations are composable.
So you, you know, grad can work with VMAT.
VMAT can work with grad.
and you don't want these to actually have to know about each other.
Right? Like, you can specify these transformations individually and then, you know, put them together whatever order you like.
So how the heck does this play well with a multiple dispatch system like we just described before with torch function? Well, remember that I said that although the order we call methods is unspecified, there is one thing that is guaranteed.
which is we are always guaranteed to run the sub class method before the parent method.
So let's say that I want to do some composition of operations, say a vmap first and then a grad.
Well, if I want to make sure that I handle the gradients before I do any v mapping, then all I need to do is make sure the gradient class subclasses the vMAP class And of course, I might do it the other way.
Right? I might want to have the v map class, sub class, the gradient class.
And so really what I want to happen in this situation is I'm actually just going to dynamically create new classes for whatever sequence of compositions I want.
So if I wanna do a v map and then grad and then a v map, well, I'll just, you know, have a v map one that inherits from grad zero that inherits from v map zero or or, you know, like, whatever.
Fortunately, Python is a very dynamic language and so it's pretty easy to allocate classes on the fly.
So you you'll have some implementation of this class, but when a user wants to actually use it, they actually have to, you know, set up this inheritance hierarchy that says, what order the transformations relate to each other? But, you know, this is not something we have to write any code for.
You can just do this for them.
on the fly by generities and classes.
And the wonderful thing about this is it says, hey, you know, functorch is this cool thing It's got all these transformations.
They're composable with each other.
And in fact, the torch function multiple dispatch mechanism or really the dispatch to Python dispatch mechanism, but they're one in the sand there.
They're literally implemented using the same code.
this mechanism is general enough to to make this work.
So we don't actually have to add any extra level or stack or anything like that to make the multiple dispatch workout in the situation.
That's pretty cool and something Richard and I didn't expect when we're trying to work out what to do in the situation.
It also answers some questions we had, which is what should happen if you have, you know, some functional transforms that aren't nested in each other and are leaking between each other.
and this would correspond to a sub class a of some parent and a sub class b of some parent, but a and b aren't really it at all.
And, you know, remember what I said about George's function, what you're supposed to do is check your types and make sure you actually understand everything that is in there.
So if you get some type that isn't related to your current class hierarchy, you're supposed to return not implemented error.
And so we'll correctly get the correct error case in this situation, which is that, well, this is not something that's implemented.
You haven't said how these two passes interact with each other.
So we're not gonna guess one way or another.
So what's the upshot? Well, Python doesn't have native multiple dispatch.
but torch function and torch dispatch dispatch to Python both implement a form of multiple dispatch for handling what happens when you pass multiple different subclasses to a function.
It's pretty simple, but very powerful and good enough to express all sorts of things including Jack style composable transformations.
That's everything I wanted to say for today.
Talk to you next time.
.
