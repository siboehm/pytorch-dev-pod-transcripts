---
layout: post
title: "Api Design Via Lexical And Dynamic Scoping"
date: 2021-07-08
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Api Design Via Lexical And Dynamic Scoping

Hello, everyone, and welcome to the Pritchard Step Podcast.
Today, I want to talk about lexical scoping, dynamic scoping, and how these Program languages concepts relate to library design in PyTorch, specifically with regards to backwards compatibility and other questions.
When I talk to people about working on Pytorch, sometimes I get questions from people who knew me before I joined the Pytorch project, as a Haskell developer working on compilers.
And they'd ask me if I was doing any programming languages stuff here in machine learning land.
And I'd always be very happy to answer people and say yes.
In fact, I use programming languages concepts all the time as a developer on the Pirate project.
And that today's podcast about lexical and dynamic scooping is an example of how I use these concepts from programming languages to reason about some actually fairly complicated API design questions that, you know, as a Python library, Pytor just to answer when we want to, you know, talk about how we're gonna design an API in question.
So to start with, I need to explain, what is lexical scoping? What is dynamic scoping? So lexical scoping is so when we talk about scoping, we're typically talking about how do we resolve what the meaning of variable it.
So when I have a function and I refer to the variable x, you know, how do I know what x is? Lexical scoping says that the value of x is whatever is lexically closest that defines the x in question.
And when I say lexically closest, I mean, imagine you're looking at the source code of your program You see the x, you your eye wanders up outside of the enclosing box until you find a block that actually defines the x variable in question.
And that definition is going to be the one that your actual use of the variable is going to point to.
In contrast, dynamic scoping is a form of scoping where the reference to x doesn't actually refer to, you know, whatever is lexically obvious.
Instead, there's a concept of an implicit, you know, global variable, if you can think of it that way, which sort of gets changed whenever you do an assignment.
So So what the value of x will be is not what you saw, you know, in the classical scoping, but in fact, whatever the color to, you know, your function set the variable to be when you when you before you actually called in the function.
So you have to look at the call stack to figure out what the value of a dynamically skilled variable is.
And so very concretely, in the Python programming language, there's no native support for dynamic scoping, but a lot of use cases that people use for context managers, you know, that's the with statement where you can with blah And then inside your inside of this block, something different happens because of the ConX Manager.
ConX Managers are very easy way to, like, implement dynamic scoping because what you do is when you enter the context manager, you set some global variable to some value When you exit, you reset it to its original value, and that's basically equivalent to having done a dynamically scoped variable assignment.
And of course, you know, regular old variable references in Python are dyslexically, if you import modules and use identification of those.
that's also done lexically.
Okay.
So up until this point, this is something that, you know, you might have gotten told about in your program languages class in undergrad.
So what the heck does this have to do with Pytronch API design? So the first thing I wanna talk about is a sort of case study in what happens when you want to change the semantics of a library or in this particular example's case, the python language itself And why, you know, whether or not you choose to do this with lexical or dynamic scoping has pretty big implications on how usable the thing is.
So here's how the case study goes.
So back in Python two, the Python developers made a bad decision.
and the bad decision they made was that they defined the slash operator to mean integer division.
This was a very understandable mistake to make because languages like c defined a single slash to be energy division, but what they found was that, like, lots of people were using Python to do, like, calculators and stuff like that.
And they'd always ask things like, what is one divided by two? And Python would helpfully or unhelpfully from your perspective.
Say zero.
and that was very unexpected.
So the Python developers decided, okay, we wanna change what the meeting of division is.
We wanna change it from integer division to true division so that if you divide one by two, you don't get zero, and second you get zero point five.
Obviously, this is BC breaking.
So how are you gonna deal with a problem like this? Well, you want some way when you have a busy breaking change to let people opt into the new behavior before it becomes mandatory, and then only at some later point in time, namely Python three, make it required.
So, you know, there's intermediate time when you can change the meaning of your program to switch from, you know, energy division into true division.
So how exactly did Python do this? Well, Python actually needed to introduce a special mechanism called a feature import to make this happen.
the way the future import worked was that there's this special module called Future, and you could say from Future Import division, and then what that would do was it would be change the meaning of all of the slashes inside your current module to go from division to true division.
Now, if you're like me and you're thinking, you know, why the heck do I have to introduce an entirely new language feature? So future is not a module.
It is like a special language feature that changes how the Python bytecode interpreter interprets your program.
And why the heck do they have to introduce this new feature? Why couldn't they just have said, well, like, something like, okay, instead of importing division from, like, the normal module import division from the, you know, like, crew division module.
The same way, you know, if I had a function, and I wanted to change the function semantics.
I could have a v one of the module and a v two of the module and I could just pick which module I imported that function from to get one version or the other.
Well, the reason they needed to do this was because the division operator actually isn't a function.
What division in python she sugars into is a call into a magic method.
And whether or not it de sugars into a call into the magic method, dive or the magic method, true dev depends precisely on your version of Python and whether or not you import future division.
So in effect, the way that the meaning of division was defined was not by lexical scoping, which and in fact, In some languages like Haskell, the meeting of division is lexically scoped.
It's provided by this prelude module that, like, is implicitly imported by your program, and that's how you tell what the meaning of division is.
That's not the case in Python.
division always desigured into one of these method indications, and method indications, well, they're not really lexically scoped or dynamically scoped.
Instead, it's a form of dynamic dispatch.
where you ask the object what the meaning of the operation should be.
And so to change the method indication that happens in this case, you actually need some actual, you know, juice from the language itself.
And so that's why the future mechanism exists.
So Piton had this problem.
The problem they had was that they wanted to change the meaning of a method invitation in a backwards and compatible way, but they had no way of letting people opt into it one by one.
So they introduced a language feature letting you change the meaning other method from one thing to another.
In Pytorch, we often want to make BC breaking changes to methods, but unfortunately for us there's no way to implement a same future style mechanism inside Pytorch.
You just you just can't do it because it requires language support and Python didn't give us language support to do this.
The best approximation for this is to have some sort of global flag which can use to toggle between the old behavior and the new behavior in question.
But notice this is very different from what future import division does.
Right? Future import division only affects the division operators inside your module.
If you import some other module that's using old school energy division, that entered your division stays the same way that it used to be.
So it's a very local.
You can reason about what the meaning of division operators is simply by just looking at the top of your file.
with a global flag, you don't actually know what the meaning is without walking up the call stack.
and looking for someone who actually set the global at some point in time.
And so we actually try very hard not to do this in PyTorch.
And the reason why we do that is gonna become clear in my second case study.
Case study two, device context manager.
To explain this case study, I have to first explain what a device context manager is.
And this is a little tricky because there's no such thing in Pytorch it is a thing that has been requested over and over again by many different users.
So here's what this hypothetical mechanism would do.
When you write hydrogen programs, You often wanna write your program in such a way that you have both CPU code and CUDA code.
So what does this look like? Well, you know, like you have your script, you wanna debug it and test it on CPU, and then at some point, you wanna rerun it again on CUDA.
And if you know anything about, like, PyTorch's API, we don't exactly make this easy to do.
You have to have to actually plan your program out and, like, explicitly, like, you know, parameters over the device in question and then, you know, toggle that with your options.
If you just sort of write, like, really plain straight line code, you're probably ending up hard coding that it operates on CPU or CUDA.
So the device conference manager is this concept that lets us you write the naive code, like allocate a bunch of tensors with no device argument, do a bunch of operations on them, and then implicitly change the meaning of the factory function so that if you you know, use this phonics manager and say, hey, set the default device to be CUDA, then whenever you do any inter calls, to the factory functions in question, they will actually produce CUDA tensors instead of CPU tensors.
So this is a decent example of dynamic scoping an off an action Right? Like, when you use one of these Connex Managers, it's not just the, like, local calls to factory functions that are in your module that would be changed from CPU to CUDA.
It's also all the inner calls to, like, all the modules you might be instantiating and everything else.
And this is kind of desirable.
Right? Because, like, one of the things that people find very annoying about how things have to be done today is you have to, like, plumb the device you want down recursively into all of the, like, creation functions that you're doing.
And in this case, This is like all of the sub modules and your modules.
By the way, we used to not actually let you plumb device down, but Joel Schlosser very recently lined in a patch to Pytorch that makes all modules take a device argument, so you can change what the device is, you know, at module construction time.
before that, you have to actually always construct your module on CPU and then move it onto the device you want it.
And that's that's kind of inefficient.
And a lot of people didn't like having to do that.
So anyway, so this device conference manager would let you change, for example, where your modules get allocated without having to actually explicitly pass in this device argument.
And so a lot of people would like this.
It would make things very convenient and we don't wanna do it.
Why don't we want to do it? Well, the reason we don't want to do it is because of the fact that it you know, actually recursively goes down and all of your calls in the call set change their semantics.
Right? This is like both a blessing and a curse.
The blessing of it is that you don't have to inter coordinate with anyone to change the device.
You just set this Connex manager and then magically the meanings of all of your factory functions change.
The curse of it is you don't have to coordinate with anyone.
So if someone writes some code that like a assumes that torch empty is just going to give you a CPU tensor because when I tested the code on my machine, it gave me a CPU tensor like, you know, what how difficult could this possibly be, that code is going to unpredictably break.
And in practice, this code unpredictably breaks because we have a janky of device Conics Managers called set default sensor type, which you can actually use to change the default sensor type from CPU to CUDA.
Please don't do this.
We really hate this function.
We want to get rid of it.
But this one, people always post forum posts being like hey, I did this thing and, like, my code, some code, library code that I'm calling doesn't work.
So the, like, problem with untype dynamic scoping is that it is a global tax on all code written in your library.
If you have primitive function calls that are modulated by some dynamic scope, by a context manager, everyone who writes library code is obligated to make sure that their code works under all possible settings of the context manager.
So in this case, whenever I write a bare torch dot empty and not bare torch dot empty device equal CPU, I'm obligated to make sure that this will work even if you do a CUDA device.
And maybe this is like possible, and maybe this is even the right trade off to make.
But historically, Piters doesn't have this requirement, and so a lot of code is not written under this assumption.
And so if you want to add a device Connex manager and you wanted to do it right, And when I say, right, I mean, like, this Connex manager actually works in, like, ninety nine percent of all the situation you use it in, you actually have to go and painstakingly audit all of your python code to make sure there's actually doing the right thing in this case.
Black.
So like, you know, dynamic scoping leads to unpredictable effects because it, like, lets you reach into code that wasn't expecting to be modulated.
Sometimes this is a good thing.
Right? Like, it saves you from having to explicitly pass arguments around.
If your e max, you know, actually, like, you love dynamic scoping because it lets makes it so easy to just set some variables and then use them later inside somewhere else without having to muck about the function signatures.
But like this implicitness also comes with a cost.
Okay.
I have one last case study and this relates to torch function and also a sort of new mechanism proposed by NumPy for handling factory functions.
So a little bit of backstory here.
So, torch function is this thing where you can write an object, you put a torch function, magic method on it, and then whenever you pass these objects into torch dot cat, torch dot add.
Any of the functions in the torch need space will actually just call this magic torch function method so that you can override the meeting of operations involving tensor sub classes.
So this is very useful and you can use it to implement all sorts of interesting tensor like objects without having to actually like you know monkey patch all of you know, Pytorch's functions to, you know, do something different in this case.
But there is a problem.
And the problem is, torch function is predicated on the idea that any given function operation takes in an actual tenses an argument.
Because the way it like does dispatch is in the very python dynamic dispatch style, we look for an object that has a torch function on it, and that's the torch function implementation we call.
So what happens when you have a function that doesn't have any tensor arguments.
And the example of that is a factory function.
Right? torch dot empty, which just takes in a list of sizes, and gives you a tensor in question.
So custom classes have a problem, which is they need to also somehow override these these factory functions, but they have no way of doing so their standard mechanism of overriding is via dynamic dispatch, but there is no dynamic dispatch in this situation.
So there are a bunch of ways to solve this problem.
As the saying goes, if the mountain won't come to Mohammed, Mohammed must go to the mountain.
So If you, you know, want dynamic dispatch and the factory function doesn't have dynamic dispatch, well, turn it into a call that does have dynamic dispatch.
So we have a bunch of functions on tensors like new, empty, and new zeros.
And, you know, you can use those in place of the good old fashioned torch function, torch factory function in the main name space.
and that will indeed work.
And then you just have to define those things in your touch function to get things going.
And this just preserves the same property right, which is that you are using the objects that are lexically in scope to do the dynamic dispatch to get to the implementation you want.
There's an elaboration on this idea, which is a is a numpy proposal at this point in time, which instead of directly, like, creating new variants of methods for tensors for all the factories functions in CRISPREDED, wrap them up into a module call.
So given a tensor, you can extract out a module that corresponds to the, you know, type of module that you would have called the factory functions on, but this one is specialized for the sub class in question.
So how what does this look like? So I've got a tensor.
I want to create a new tensor.
So on this sensor, I call the module processor, which gives me a torch module, something that looks like torch, so it's got empty and it's got ones and it got zeros on it.
But this module is special because if I call zeros on this module, I will actually get a tensor that is of the same sub class as whatever my original tester that I got this module out from from the beginning.
So same same idea.
Right? Use the lexically scoped values to get out the the the module and then do the dynamics especially on the module itself.
So you just don't have to, like, shove everything into the method name space.
Of course, there's another way to do this, and that's using a context manager.
And this is actually more likely than you would might think.
So in previous podcasts, I've talked about Fungtorch, a method for doing, you know, functional transformations on Pytorch programs.
And in Funk Forch, there's a very natural place where a context manager would be applied, and that's when you use one of the higher order commonators like vmap to actually do an operation on a tensor.
So when I enter the v map, what I'm effectively gonna do is I'm going to basically turn on the v mapping as And what that also means is that I might very reasonably want to override the behavior of all the factory functions as well implicitly when I do this.
And this is actually very natural.
And in fact, in Jack's, this concept is called Omni staging, where in previously Jack's only did data dependent control flow, but at some point in the future, they realized, hey, actually, it's really useful to be able to, you know, override the behavior of these free functions.
And so, you know, let me just go ahead and do that.
And so that's that's called Omni staging in JAKs.
So which of these is the right thing? Well, if we look back to our previous case study on device connex manager, PYDERGE said, hey, you know, we want explicitness.
We don't want we've got all this code that's been written already that doesn't think that you're going to, like, change the meaning of things under your feet.
So, like, you know, let's just make sure that you keep doing things explicitly.
And so we're we're we don't really wanna add this Connexus manager.
But then when we look at this, you know, touch function module case, you know, there is a solution that you can do to, you know, stay with the lexical attitude, which honestly is Pytorch's attitude, but you can also see that there is a lot of merit to doing the dynamic scooping.
And these problems of backwards compatibility don't, you know, they're they're not as pressing because although you might have not had written your code so that it works correctly under CPU or CUDA, with vmap, well, you know, you're you're explicitly asking for vmap in this case.
So one is you're you're probably gonna like make sure all the code you're calling is stuff that works correctly in this case.
And two is that v m m actually, you know, is very carefully written so that, like, the code on the inside looks exactly like you're doing a single example case.
So it really is supposed to work even if you like change out the semantics of everything.
It's just you're just, you know, adding these batch dimensions in a way that, like, is your code should be indifferent.
too.
So what's the right answer? Well, I don't really know.
When I talk to people and they ask me for device connex manager, you know, I used to call over Greg and Greg were like, no, we're not gonna do this because everyone's code is not gonna work in this case.
Well, maybe.
If you're willing to put in the work to make this all work correctly and all the library and all the ecosystem, I think you know, some dynamic scoping might actually be pretty helpful, but there's a lot of work.
And I want to see this work actually, you know, have an honest attempt for this.
That's everything I wanted to talk about for today.
Talk to you next time.
.
