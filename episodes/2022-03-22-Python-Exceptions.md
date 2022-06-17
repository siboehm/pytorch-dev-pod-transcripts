---
layout: post
title: "Python Exceptions"
date: 2022-03-22
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Python Exceptions

Hello, everyone, and welcome to the Pytorch Tip podcast.
Today, I want to talk about exceptional in handling in Pytorch.
Specifically, how we handle the boundary between python and c plus plus in PyTorch.
So where to start? Well, let's start off by talking a little bit about c plus plus exceptions.
love them, hate them.
They're kind of a very interesting language feature.
So c plus plus exceptions are based off of the idea that hey, We want a mechanism for doing error handling in the c plus plus error language, which doesn't cost anything when there is no exception.
And as a result, Exceptions have a very interesting performance characteristic, which is that when your code goes well, the exception handling logic doesn't really cost you anything besides the binary size.
But when you do raise an exception, then things go very, very slowly.
There is a very slow stacked unwind ending process that uses some Lookaside tables to figure out how far you need to go, you to actually use this table, you need to take it a lock, It's very very slow.
And because of this and also because of the binary sized bloat that's associated with exception handling, a lot of environments, eG mobile, don't really want to compile with exceptions turned on.
And so, you know, you don't really wanna use exceptions most of the time when you're writing normal c plus plus code.
But of course, there are some situations where exceptions are appropriate, and I think Pytorch's use of exceptions is quite appropriate.
So PyTorch specifically uses exceptions whenever there's some sort of, I'd say, user error.
So, you know, you add two chances together, but their shapes mismatch, we need to raise an error to the user, we do an exception in the situation.
It'll be a really big pain to try to manually pipe back the error status through all of our code in this sort of exceptional situation.
Now if you're a go language developer, that's the sort of thing that you're used to doing.
Right? Like hey, you know, explicit is better than implicit, but these really are edge cases and most of the time you're not gonna hit them and it wouldn't be a good thing in our code to actually have to explicitly deal with all the error handling all the time.
And plus, it wouldn't look very pythonic.
And as I've mentioned in earlier podcasts, you know, we're all about writing c plus plus code.
That looks a lot like the python code you wanna do.
So these exceptions, they don't have happen normally.
Please don't write code that raises exceptions and expects to catch them.
Right? The point of the exception is just so that we can bubble it up to Python, turn it into a regular Python exception, and, you know, usually this will fail a user's program.
But if, you know, there's something that they actually want to do with the exception, like say they're in a rep pool and so you can just bring back control to the user, while we wanna give the user the ability to do that in that situation.
This does sometimes cause some problems for example, we had a bunch of linear algebra operations that when the matrices were ill conditioned, they raised an exception.
And some people, you know, caught those exceptions because they knew that they could use some other algorithm in these situations, and this was very very slow.
And we actually added extra APIs for getting back the error status in those cases as a bullion, so not raising exception in this case.
So exceptions, therefore, really exceptional things don't use them for, you know, things that you expect to happen when your code is running normally.
Alright.
So we're using c plus plus exceptions to handle things inside, you know, the bowels of the c plus plus Pie torch.
But remember that once we hit the Python c plus plus language boundary, we actually need them to be treated as Python exceptions.
and now see Python, the Python implementation that most people use, is not implemented in c plus plus.
It's implemented in c And as such, it actually has no idea what is going on with c plus plus exceptions.
So you actually have to do some conversion.
So the convention in Python for handling exceptions and because at c, you do have to do it for everything explicitly and in c Python on source code, it does handle everything explicitly, is you are obligated to check the return types of all functions you call.
And normally, these functions will return Py object pointers.
But if a error was set, if some sort of Pyzone exception was set, The object that will be returned is in fact a null pointer.
And there is some extra state, you know, off to the side, some global state which gets populated with the exception info in this situation.
Global error reporting state is very, you know, nineties error no style reporting.
But remember, Python has a global interpreter lock.
So you're not really at risk of some other, you know, thread stomping over your exception state in a situation.
So if you return a null pointer, that means an error has happened and there's, you know, you're supposed to go ahead and propagate this null pointer up until some point where ception handling can actually happen.
So to interoperate between c plus plus exceptions and Python exceptions, It seems fairly simple.
What we need to do is we need to catch the c plus plus exception before we go to the python boundary then we need to go ahead and, you know, take out this exception, look at it, convert it into a Python exception that we can also, you know, save to the global state saying that there's a Python exception.
And then we just need to return null pointer in that situation.
Seems easy enough.
Right? Well, you have to actually remember to call the macro that actually does this.
So in a kind of poorly named set of macros, we probably should rename these macros.
They're called handle error and end handle teach error.
So when you're writing Python binding code, you need to make sure that you, you know, start off with a handle teach error, which set up this try catch block and then an end hand an old t h error, which will, you know, sort of, handle the end of the try catch block.
including the catching exception, turning it into a Python error, and then returning null pointers, so see Python knows what's up.
but wait, there's more.
So we also use pibbe in eleven to do some python binding inside of our source code.
and Pi nine eleven has a different convention than c Python.
C Python says return an null pointer and we'll handle it.
Pi nine eleven says we're a c plus plus library.
We like exceptions too.
And so in fact, Pi by eleven knows how to deal with exceptions.
And in fact, we install a handler handler.
Thanks Peter Bell for adding this, which will know how to automatically convert exceptions into the form that is expected by the c Python interpreter.
So you don't have to use handle TH error when you're doing PYBEIND, findings, question mark.
Actually, answer is no, you do.
You still have to use them, but that's another story, which we will talk about in the second part of this podcast.
But yeah, so Pi nine eleven has a different convention.
And if you've actually gone ahead and set the Pi on error state already, there is a special exception in pibyn11 that says error already set, and that's the one that you can throw to have five eleven say, oh, I see.
You've already set the info, so I'm not gonna do anything, return an null pointer in that situation.
So Now there it's not obvious that c plus plus exceptions should map to Python exceptions, but we have a bunch of sort of pre canned exceptions they're all defined in c ten exceptions dot h, like not implemented error and similar things like type error, And so if you want your c plus plus exception to turn into a particular Python error handling class, just make sure you use the correct you know, error class or are there a number of macros that also let you, you know, specialize what type you get in that situation? Alright.
So if that was everything that handle t h error did, I'd be done with this podcast in eight minutes, but it's not.
There's actually more.
So exceptions are pretty nice and, you know, we like using them a lot to handle error cases.
And there's something else that's pretty nice, which is warnings.
We love warnings.
probably a little too much.
We probably PyTorch has just, you know, sort of grown warnings over time and, like, people have stopped reading them and it's add and we should get the warnings to be less chatty.
That's a topic for another time.
So, warnings are pretty useful because, hey, sometimes people are doing things that are kind of bad and we don't wanna error on them, but we do wanna let people know that, you know, something bad is up.
Like, for example, using a function that we've deprecated, and plan to remove in the future.
And a lot of this code only actually gets exercised in c plus plus.
So we want some way of reporting warnings.
Now, it's easy enough to, you know, write a c plus plus warning function that just prints some stuff out to standard error.
But similar to how exceptions have their own handling in Python, right, with the, you know, good old fashioned Python exceptions, warnings also have hand ling in Python.
There's a warnings module.
There's a concept of warnings filters and warnings handlers.
And it would be nice if the warnings raised by Pytruch interoperated with this framework, and they do.
So what we have is we have a way of mapping c plus plus loss warnings into Python warnings.
So when you use the torch warn macro, which is the way of, you know, basically raising warning from c plus plus code, What it will actually do is it will convert it into a Python warning and, you know, send it off so that you can, for example, ignore it.
as as these things typically do when you are actually dealing with it in your Python code.
Now it used to be implemented such that we would take out the global interpreter lock because remember when we're in c plus plus code, we released the global interpreter lock so that other threads can keep going.
And so we would have to reacquire it and then, you know, fiddle around with python state to actually raise the warning.
But this sometimes caused dead locks.
So, Alvin, a few years ago, submitted a patch to make this better.
And the idea is that, well, there isn't really any point in reporting the warnings to user land until we actually, you know, get back to the python interpreter.
So we can basically defer all of the warnings we want to raise until we you know, go back to Python.
In fact, the c Python API has a dedicated function for doing this sort of thing.
It's basically add a callback, which when the next and the Gill is acquired, we'll do these callbacks.
And and this thing is protected by its own very tiny lock so you can take it out without fear of deadlocking the gilt.
But we didn't use that for this particular mechanism.
Instead, we have our own little buffer that warnings get written to, and then we have some way of propagating to Python when we return.
And how does this work? Well, we piggy backed on top of the existing handle t h error macro.
So how do you get some code to run when you're exiting a code block well in c plus plus the way to do that is r a i i.
So you allocate an object on the stack.
And then when, you know, you're exiting the scope, by returning or by raising exception, then the destructor for this object will get called.
All happy.
Right? well, no.
So I mentioned that when we have exceptions in c plus plus, we turn them into python exceptions.
And so at the point in time, when we're handling the warnings, basically feeding them into the Python interpreter, we might have an active exception at this point in time.
And now there's a problem.
When you print a warning to when you when you put a warning into python's warning system, you actually might be running arbitrary code.
Why? Well, you need to actually construct the warning object and there's also some handlers which, you know, might actually just go ahead and process the warning immediately when you do it.
And all of this code can raise errors.
And so what do you do if you are raising an exception and the unwinding code also tries to raise an exception at the same time.
While c plus plus has an answer for this, it's, you know, abort your program turn immediately, you know, unseriously killing everything that's going on.
Well, that's kind of bad and we don't really want to do that.
Right? We wanna make sure we always get to Python in this case.
So if that happens, then you have to basically not run the warning handlers if there's an exception being risen because, you know, you are not going to be able to deal with another exception being raised at that point in time.
And so the way we do this is just if that happens, we don't actually gave you the warnings in Python.
We'll just print them to Stidder and c plus plus.
And they and they vanish in it either.
Well, you can still see them in standard error, but they won't be available in Python.
on.
And that's pretty reasonable because this only happens when you were raising an error anyway.
And remember, those are exceptional situations, and so, you know, you really shouldn't be doing that.
Well, there is one subtle point though, which is that remember how I said that if you set a Python error you know, the global flags inside Python, and then you return null pointer.
Python will know what to do with that.
that technically worked before even if you were using the handle teach error macros.
But now you're not allowed to do that.
Because if you if you are just returning an old pointer, then the warnings handler will run and it won't know if there is a Python error or not and it might accidentally try to raise an error again and that's that's bad.
Okay.
So that's it for error handling.
So if you don't remember anything from this podcast, remember to put your handle t h errors and end handle t h errors around your bindings, otherwise your exceptions won't work correctly.
Or use Pipeline eleven.
But if you're using Pipeline eleven, you still probably wanna use these macros or the nifty, you know, wrap warning handler function, which I will post in the podcast liner notes if you need to look it up, which is just a nicer way of doing the same thing.
without using macros.
Make sure you do that because otherwise if you raise warnings in your code, those won't work either.
And yes, this is probably too hard to remember and we probably should have a lint about this and we don't really have a good linting framework.
That's a good topic for another time.
Alright.
That's everything I wanted to say for today.
Talk to you next time.
.
