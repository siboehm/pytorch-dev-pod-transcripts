---
layout: post
title: "Mobile Selective Build"
date: 2021-06-06
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Mobile Selective Build

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk about mobile selective build.
Pytorch is a project that is trying to do a lot of things.
And one of the more unconventional things that the project tries to do is we use the same code base that you use for doing your, you know, good old fashioned Python training loops in Python on your regular desktop.
And we also use this code base for actually deploying Pirate's mobile models on mobile, like, so that you can run some, you know, image model on your phone and get a result back without actually having to go back all the way to a server.
So this is kind of crazy actually because mobile is a completely different universe than server side programming.
And There's one particular aspect of it that I wanna talk about today, which is selective build.
Namely, the fact that when you are Writing applications that go on mobile binary size is really, really, really important.
On server binary size isn't that important It is kind of important because if your binary gets too big, like say four gigabytes big, then a lot of tools like the debuggers stop working.
but it takes a lot of code to get to four gigabytes.
On mobile, this isn't really the case.
You really, really want your app to be as small as possible because you've got people who are downloading your app on really, you know, shitty cell phone connections.
And if your app takes a lot of binary, then they're not happy.
And without really having any sort of clamp on the binary size, the easy thing for a software project to do is just keep going and going and binary and so there's very stringent restrictions about binary size.
People will yell at you if binary size increases too much and it's in this context that Pytorch designed selected build.
So what is selected build in Pytorch? Well, this is a concept that hey, Pytorch comes with a lot of operators, right, a lot of support for many, many different operations.
And half the time, you're not using even half of these operators for any given model.
Right? If you're like doing a resnet, oh, so old fashioned.
But if you're doing a resnet, there'd probably only be twenty so or so operations that you act need out of Pytorch's, you know, more than a thousand operators.
So what's the idea? If you are shipping some models to mobile and you know what the set of models you want to do are, well, don't ship all the operators.
Ship only the operators that you actually need to run on mobile.
and you'll get big binary sized savings and everyone will love you.
And also, all of the people who are, you know, frantically working on adding new functionalities of Pytorch they don't have to worry about going over some binary size limit because all that stuff isn't actually going to be used.
Now ordinarily, when you are building some application for mobile, typically the way you do it is you build everything statically and you statically link everything together and static linking has this interesting property which is that we know exactly what is being used inside a statically linked application.
So if a function is not being used, we can actually just prune it away.
And all of linkers will do that automatically in that situation.
You can't do this by the way for a dynamic library because a dynamic library offers a public API and anyone else Even people you know nothing about could make a use of any of the exposed functions in your dynamic API.
So usually everything has to be put in.
So if a static library can be done this way, why doesn't, you know, elimination of operators that you don't need happen automatically in PyTorch? Well, there's two reasons.
So one is that when we run when we run models on mobile, we're running them via an interpreter.
Either the torch grip interpreter or the light interpreter, which is a sort of pared down version of Torus Script that has less support but, you know, is smaller in binary size and runs a little faster.
So when you have an interpreter, one of the things in the interpreter loop that you need to do is you need to, you know, look at your op code which says, hey, run this operator and have a giant switch statement for all the operators that you understand and, you know, have a call to each of them.
And, obviously, static linking isn't going to know that, well, this particular branch, which is doing some, you know, niche activation or whatever, isn't actually ever gonna be used by your mobile.
Because it can't.
No.
There's no way for it to know.
So we need to tell the interpreter, hey, you know, these ops you don't need to compile in.
You can't get it automatically with static linking.
But let's say you wrote your model directly in c plus plus, which is something you can do and you could actually use to deploy models.
Although most people don't because it's a pain in the ass to update native code on mobile because you have to, you know, build an entirely new version of your app.
It's much easier to just push an on the wire update.
for some data that just is your, you know, serialized model.
But let's say you did do that.
Hypothetically static linking should get you what you want in this case.
Right? Well, not quite either.
So in PyTorch, we use this operator registration mechanism to make it possible for people to sort of insert in, it's like a form of dependency injection.
Like, if you load up the Lip George Kuta library, then all calls to torch dot ad suddenly have the ability to call into CUDA as long as they're passed by CUDA tensor.
And this is done via dynamic dispatch.
And the important thing is that in order for to make this dynamic dispatch work, we have to register an implementation of the operator at library loading time.
And what happens when you do that? Well, that's a static initializer in the library.
And once again, the compiler cannot eliminate this because it doesn't know if this arbitrary piece of code that gets run at library startup might actually you know, do something important that you can't dispense with.
So, okay.
By the way, that's why you need, like, whole archive if you're linking against pipe or it's statically.
because otherwise, they'll just drop all the static initializers if nothing in the object file in question is referenced.
It's it's pretty nuts so, but you know, that that's the way it is.
Okay.
So we need a way to actually figure out what operators that our model needs.
and then apply this to a build of PyTorch so that we don't we don't actually send them when we're building the application for mobile.
Okay.
So let's take these in two steps.
So first, what operators does our model need? So if I have a torch script model, my term script model is serialized in some machine readable form.
And so at the first level, it's really easy to figure out what operators' model needs.
Right? Like, we just go to this serialized format.
And for every operator call in it, we just say, okay.
Well, I see an ad, so I need ad, and then, oh, I see a convolution.
So I need convolution, etcetera.
easy to get a list of operators that the model needs.
But there's a problem with this, which is what if your operator uses other operators? And this is really really common in PyTorch because we have a lot of, like, really small cheap operations that you can use to sort of massage things into the correct form.
Like, viewing and reshaping, and many many operators use this.
And so if you are doing one of these things, well, you also need to be able to track what those uses are.
You need some sort of dependency graph from operators to operators.
So how is this done? Well, the way we do this is we actually have a LLVM based static analysis.
What you do is you take PyTorch, you compile it with Clang producing LLVM bit code for all the object files, and then our static analysis goes through and looks through all of these all of the bit code looks for things that look like operator definitions.
They're easy to find because there's a specific API call you use to register to the operator.
So it just looks for instances of that API call.
And then it, you know, spiders that code until it finds all the dispatcher calls, which mean that, hey, I have a dynamic dependency on some operator and then generates that into a YAML.
That's pretty interesting.
Most people don't wanna compile PyTorch with LLVM bit code to actually get this analysis graph.
So we also have the Yamal checked in for an easy kick start if you don't actually wanna, you know, run this pass.
By the way, this pass is supposed to be updated by a bot, but the last I was I was checking for this podcast.
The last time it was updated was February this year.
So you know, if you're running into a problem with the open source mobile selected build like something's missing and it shouldn't be, just rebuild from scratch.
The instructions are there.
It's pretty simple.
I'll also link it in the episode notes for this podcast.
By the way, there's another way to get the way things your ops needs, which is some sort of dynamic tracing.
And we actually debated a lot when we were trying to decide what to do for figuring out what upstream memory needs.
So what how does dynamic tracing work? Well, Instead of trying to statically read out the operators your model needs by looking at the TorusGrid model, just run the TorusGrid model.
And when you run it, you're gonna hit a bunch of operations and record what operations you see, and then that gives you exactly the set of operators you need.
So no need for, you know, this dependency graph analysis.
life is easy when you're dynamic.
Of course, there's a problem with this.
Right? Which is you need representative inputs for your model.
And well, maybe that's not a big deal if you're like deploying these models because you want the representative inputs any way to test that the model doesn't crash.
but if there's say control flow in your model, then a single representative input might not actually cover everything.
So you need to make sure you you actually fully cover it's like a code coverage problem.
Right? You need to actually cover every operator that's actually used to make sure that you've got in everything.
Okay.
So that's how you get all the ops your model needs.
How do we actually apply this to a build of PyTorch? So as I said, static linking doesn't let us, you know, do this automatically.
So what we actually have to do you have to, you know, take these operator registrations or other things that would otherwise force the compiler to include a code in question and make sure that we have a way to say, okay, don't do that when we don't need it.
So a lot of operator registrations are done by a cogeneration.
See one of my previous podcasts.
So In that case, it's very simple.
We actually just feed in the YAML file that says all the operators we need to our cogeneration and the cogeneration says, oh, you know, the site to build says that I don't need this operator.
So I'm just not gonna generate a registration call.
And if I don't generate a registration call, then the code that it calls is now dead because there's nothing actually calling it, and then they'll get pruned away by the static linker.
No problem.
Unfortunately, there are some registration calls that don't actually get generated by co gen.
They're just done manually via our, you know, very nice and intuitive m dot deaf or m dot ample syntax.
So for this, we have a very clever scheme which is called the selective name macro.
The basic idea behind this macro is that when you build PyTorch, we also dump all of the operators that are supported into a constex per string And so we actually have this constexpr function, which can basically take in an operator name and say, hey, is this included in the giant comma separated context list of, you know, all the operators that are allowed or not.
And but the selected name macro does is it just applies this constexpr function to the name that you are registering.
So you you wrap selective name around the name you wanna register.
And if it is in the context for a list, you let it go through.
No problem.
And if it's not, you generate a, you know, basically a dummy type that says, hey, don't actually do this registration.
because this all happens in compile time, then the compiler knows, oh, okay.
Now I'm just not gonna generate any code for this at all.
We had to do this a little especially because n c plus plus you actually pass strings to templates directly.
So, you know, we have to make sure this gets all resolved into a bullion, which we can then pass into a template.
There's one last detail which is actually pretty important when you're trying to understand how the selected build system works.
which is how this integrates into your built system.
So in C make, everything is fine.
You just do a C make build of Pytorch with the particular operators that you wanted to ship, and then, you know, there's no problem.
But at Facebook, we actually have multiple apps and all all these apps wanna use Facebook.
And so we actually have this problem which is that we want different sets of allowed operators depending on which app we're doing.
And the build system we use at Facebook, namely Buck, has a constraint, which is that you're only allowed to have one copy of any given library at any given time in the build system.
And this is just to make sure people aren't like doing some sort of node j s style disaster where there's like a bazillion copies of the same dependency everywhere.
But that's a problem for us.
Right? Because, you know, there's only one Pytosh library, but each of the apps wants a different version of the Pytosh library in this situation.
So what do we do? Well, we cheat.
We actually generate multiple copies of the Pytros library for each version of the app that we need in this situation.
And we don't we don't generate a copy of everything just the relevant parts that actually contain the operators.
This used to be just some glue code, which did the registrations.
So it was a very small bit of code that we, like, had to recompile for everything.
but we've actually expanded this to recompile all of PyTorch because as I said, we want selective d types and d types are like sort of coded into the operators themselves.
So there's no like registration mechanism we can use for d types.
We have to handle this actually by recompiling the kernels in question.
There's a kind of funny alternate universe where instead of like recompiling the entire library for the sets of operators you wanna do, you could also just modularize library.
So they have you have one library for convolution, another library for add, another library for sub, etcetera, etcetera, etcetera.
So isn't that like the, you know, good software engineering way to, like, you know, deal with the system? And then you only depend on the libraries you need.
Well, yes, this kinda works.
And actually, Cafe two used to do this.
And there's a problem, which is that One, building libraries takes a while.
Right? because you have to link them.
So it's like takes a minute apart.
So that would be really really slow.
And second, well, people just don't write code this way.
They don't generate a thousand libraries for a thousand really small pieces of functionality and then, you know, mix and match them for what you actually wanna do.
And a lot of the ecosystem is not set up to do this properly.
So for example, we have to load iOS applications into x code to actually, you know, work on them.
if we actually generated library for every operator, it would crash x code because there's just too many libraries.
So, you know, yeah, don't don't do no JS style stuff in in mobile.
One final thing I wanna say So the sudden build for mobile is intended to be something that you don't really have to worry about if you're developing PyTorch, but sometimes it rears its head.
And the most common situation at Reza's head is you're working on a kernel.
You modify some of its implementation details so that it's calling some new operator And then some guy comes to you and says, hey, my random mobile, like, application start working.
And that's usually be because there's some Yamal somewhere that describes the set of optimal needs and it's out of date.
Right? Because you changed what the dependency structure of the model is And so now there's a different way there's a different set of operators that are needed.
And you have to tell the Yamal file, hey, this is a new thing.
You have to rerun the analysis pass.
A lot of these things are checked in for better or for worse.
Fortunately, it's really easy to rejoin with this Yamal files, and also the PyTorch edge developers are very friendly and very willing to help in these situations.
So you can just reach out and, you know, learn how to do it.
And there's also ample documentation internally for this sort of workflow.
Okay.
That's everything I wanna talk about today.
Talk to you next time.
.
