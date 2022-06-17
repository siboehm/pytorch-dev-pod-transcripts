---
layout: post
title: "Default Arguments"
date: 2021-06-23
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Default Arguments

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I wanna do something a little different.
Normally, when I do these podcasts, I talk about various aspects of Pytorch.
But this time, I wanna talk instead about a general programming languages concept in this case, namely default arguments, which, you know, sort of, is interesting its own right.
has interesting implications on various design problems that we have in the Pytruch library.
So to start, I'm I have to explain what a default argument is Chances are you you know what default arguments are, but I just wanna spell it out for a moment.
So default arguments are a feature in many programming languages whereby when you're writing a function, you have a bunch of arguments and some of the arguments don't have to be specified.
Instead, they have defaults.
Usually specified at the definition site and those defaults, if you don't specify the argument, the argument takes on the value from that default.
So Python supports default arguments it's, in fact, the only way to implement overloads in Python without using some sort of, like, fancy decorator business or anything like that.
The point of default arguments in language design is one, they're a very compact way of defining overloads.
Right? Like, so normally, if you wanna write a function, that has that can take three arguments, four arguments or five arguments, you have to write each of these overloads as separate definitions and implement them differently.
But default arguments say, oh, I can define the four overload version in terms of the five just by filling in the last argument with the default and then calling the five in that case.
So it lets you write one definition instead of end definitions.
Another important function for default arguments is they give you the ability to retroactively add more functionality to an API without breaking backwards compatibility with any clients who were using it before.
Right? So if you have a bunch of people who are calling the function with two arguments, you wanna add a third optional argument.
Well, if you make it a required argument, all those call sites break.
But if it's defaulted, if it's optional, then all those original call sites keep working, and they will just, you know, do what the default functionality is in that case.
and we use this feature a ton in Pytorch because every once in a while we want to add a few more knobs to, you know, some sort of function or other And, you know, if we had to create a new function every time we wanted to do this, we'd have tons and tons of functions and would be hard to find things.
So being able to add extra features onto the existing names because the names are a limited name space is very useful for us.
And this is inextricably tied to another language feature that Python supports, which is keyword arguments.
Right? So keyword arguments that you add new functionality and also do so in a non positional way.
So you, like, can just say explicitly what variable name you want to specify for the argument in question.
So to put it in other words, default arguments are a way of canonicalizing multiple overloads to the maximumarity a function may take.
Let's unpack that statement.
So what do I mean by canonicalization well, you know, to colonize means to put something in a form that is the same no matter how you express it.
Right? So when I take a two argument function and then fill out its default arguments, so it's five argument, I'm doing a canonicalization process I'm canonicalizing all my function calls so that no matter how many arguments they took, I always see them with five arguments.
and Arity is the technical term for a number of arguments a function takes.
So max Arity just means that we always colonize to the maximum number of arguments.
I'm emphasizing this because I'm gonna flip this around later in the podcast.
One more thing I wanna say is that default arguments imply overloads.
But overloads are a more general concept than default arguments.
Right? So like in c plus plus, you can define overloads manually.
Anything you could have written using default arguments, you can also do using overloads.
And in some languages, there's just no overloading at all So it's not a question of do they support default arguments or not? No.
No.
No.
No.
There's just no overloading.
And common reasons why people don't want to put overloads in their language or it makes type inference more complicated or, you know, it just sort of makes it lets people write code that might be too complicated, you know, is too overloaded.
So Haskell is a good example of a language that doesn't have overloading, but another one is Golang.
They also don't believe in overloads.
So default arguments are pretty handy.
We use them a lot in PyTorch, but we've also had a lot of trouble that has come from, you know, sort of taking the very python centric approach to default arguments.
So I wanna explain a few of the problems that we've run into over the years.
So one problem, which I have also mentioned in the serialization podcast, is that we have a forward compatibility problem with towards script serialization.
Okay.
So what's going on here? So when you write a PyTorch model and you torch script it into some sort of, like, representation for the IR in question.
When we serialize it, in old versions of PyTorch, we serialize these with all of the default arguments written out.
Right? Why do all the default arguments show up when you serialize? Well, remember that default arguments are canonicalization mechanism.
Right? So by the time we've gotten to the Tourscript IR, there's actually only one representation for any given call.
And that's the canonical form, which is the maxarity, as I said.
And when we serialize and we look at these function calls, while they've already been canonicalized to max form, So the simplest and easiest thing to do is to serialize them back out into the Torbjorn model format with all of the arguments because that's what the input i r had.
And this is a Ford's compatibility problem.
So Ford's compatibility refers to when you do something, does it work with previous versions of the software? And so the problem is if I add a new optional argument to a function, I will start serializing code that has this argument explicitly filled in.
But all versions of Pytruich won't have that argument, and they will choke when that new argument shows up.
So this is a sense in which, like, canonicalizing in this way, like reduces the amount of, you know, implementations in the back end that are possible.
Right? Like previously, if I had a function that could only deal with four arguments, as long as I pass it only four arguments, it would be fine.
But once I pass up this fifth argument, even if it's the default value, even if it would have behaved exactly the same way as the four argument version, I'm stuck because, you know, the the the back end, the server doesn't actually know that this is the case.
So the telescope's realization FC problem is one manifestation of troubles with default arguments, but actually there are other manifestations as well.
So let's talk a little bit about x lite and back end extensibility.
So back end extensibility says that you can define your own custom device on Pytorch like XLA or, you know, anything else, and then define implementations for all the operators in Pytorch.
and how do you define these implementations? Well, you define the maxarity implementation for any given function.
So if function has a bunch of defaults in it, you have to write a function that handles all of the defaults.
So what do you think happens when I add a new defaulted argument to a function in Pyturg? well, the back end extensions all break.
Right? So, like, if you're if you're remembering that, like, x l a, you know, things break in x l a, Well, that's because usually people are adding new things to the schema.
And because our current API, if we're doing back to an extension, requires you to implement once again the maxarity implementation.
Whenever this happens, someone has to go to x l a and add support for the new argument in question.
And they can even be just as simple as, like, testing if it's the default value and if it's not raising error.
But they have to intervene because the APIs require you to provide the max thing.
So it's it's strictly BC breaking from the perspective of the server.
One last example is let's say that you're in FX.
So FX is our transformation framework in Pytorch and you want to, you know, do a bunch of ad hoc transformations on your model to, like, get it into some other form.
Maybe you wanna shard it or, you know, you wanna view some things.
Very common feature for FX passes is they're very specific, they're very domain specific.
So you're not, like, trying to write a general pass to the work in all cases.
There's probably some particular use case you're looking at.
And you're you're gonna ignore most operators and only the few operators you really care about are the ones you're gonna do.
And so if you're doing one of these FX passes and previously an operator like had two arguments, you might write your FX plus under the assumption that when this operator shows up in your IR, there's gonna be two arguments in it.
And once again, if I add a new optional argument to it, And so now it gets canonicalized in the IR to have three arguments.
Well, oh, no.
All of your old, you know, code doing this transformation path is broken because, well, it wasn't it didn't know how to it doesn't know how to deal with this third argument.
Even though this Third argument, most of the time, if you've used the default values, would have been semantically equivalent to the two argument version.
So all three of these examples are the same side.
They're they're just, you know, different sides of my three sided dice, which isn't a thing, but Right? It's the problem is default arguments are really good for maintaining client compatibility.
They're really good for maintaining compatibility with the caller of code.
but they're really bad at maintaining compatibility with the so called server, the implementer of the code.
Right? Because under the sort of python model, you have to deal with all the arguments because immediately what happens once you have called one of these defaulted functions is you get all the arguments and now you're expected to handle them all.
Okay.
So how did JavaScript solve the serialization f c problem? I think I claimed in a previous podcast that it wasn't solved.
It actually is solved now.
The the fix landed within the last few months.
And they did a very, very cool and useful hack.
And this is kinonicalization to low airity.
So what do I mean by that? So imagine that, you know, I'm doing one of these calls, so I I might go through Python and Python is gonna go ahead and canonicalize to maxarity because I don't have a choice that's how default arguments work in Python.
and it's gonna went my way through my system.
And, eventually, I'm gonna get to serialization time and I'm gonna be like, hey, I need to write out some code that represents this argument in question.
What should I write out? And so what canonicalization to Laura already says is, hey, let's look at the arguments and see if they're actually the defaults.
And sometimes, you know, they're gonna be dynamically computed, so I'm not gonna know.
So I I I have to actually, you know, pass in a real tensor a lot of times they're constants and so I can just compare the constant against what the default is supposed to be and oh look at that.
The last two arguments are actually the defaults.
And so in this situation, what I will do is I will chop off those arguments and serialize the lowest Arity version of the function that accurately describes semantics of the calling question.
Right? So basically, like, look at the suffix of arguments, all the defaults get dropped, and there you go.
And so you can see that this solves the FC problem because even if I, you know, call some code and it fills in the default that was new and my new version of Pytorch and then, you know, wasn't supported by the old version.
As long as it's the default value, Pitarch will know to remove that argument in the end, and then I will end up with a, you know, lowerarity function that my old extension knows how to do.
And we can actually play this trick again for, like, all the other cases.
We haven't yet, but, like, one of the reasons I'm recruiting this podcast is a recent realization that we should apply this same technique and as other cases.
So, like, if you're a back and extender, And, you know, you have written a function that only knows how to deal with some amount of arguments.
Ideally, we would, you know, chop off the defaults so that your code would still work in that situation.
And of course, this is kind of hard to do in c plus plus, but we are working on this new Python back end extensibility mechanism called torch dispatch.
And there, we actually can do this and it's not too hard to do and we should do it.
So there you have it.
Right? So default arguments are this way of canonicalizing your function calls to their max r d form.
But maxarity is bad for servers.
Right? It's good for clients.
It's bad for servers.
And so what you want to do instead is if you are transitioning back across the sort of abstraction boundary to the extensibility point on on the back end, a good technique to apply in this case.
is to reeconomics back to lower already chopping off the default arguments that are not necessary.
There's like a sort of meta lesson that I took from Right? Which is that, you know, we designed our API, our Jitskema API off of Python language design because Pytorch from the very beginning was a Python language library, and so we assumed that overloading was possible.
We assumed all these things.
And doing that, you know, gave us a very nice easy to use API for users, and it was kind of bad for backwards compatibility and forwards compatibility.
Right? And, you know, when a lot of people complain about how Golang, like, doesn't give you any toys and, like, it doesn't like we do overloading and, you know, it's really ugly writing code in Golar.
But I kinda do think Golar has a point.
Right? Which is that it's similar to do back as competitively and force competitively.
If you don't have any of this stuff.
Right? Because if you don't have default arguments, then, like, if you wanna add a new version of the function that has another argument, just gonna make a new function for that.
You just don't run into any of these problems.
Right? Like the language design of Python puts you into a situation where you have to remember to reeconomicalize the lowerarity.
But, like, if you have just separate functions, you don't have to deal with that.
But of course, doing it this way is ugly and verbose.
And so at High Churchland, we want the best of both worlds.
So, you know, we need to strike a balance and the hack of, you know, going to lower priority is a pretty good balance in my opinion.
One last thing, which is that my PhD thesis was basically on exactly this topic.
And I was very happy I didn't have to worry about overloads because Haskell doesn't have overloads and, like, once again, like, very easy.
And we we had to deal with, like, other stuff like type classes.
Type classes.
Oh my god.
Such a such a pain.
Alright.
So that's it for today.
I want to explicitly credit Dimitrijula Kove.
We had a chat before this podcast recording and he helped me solidify some of the things that I wanna say here.
That's all I wanted to say for today.
Talk to you next time.
.
