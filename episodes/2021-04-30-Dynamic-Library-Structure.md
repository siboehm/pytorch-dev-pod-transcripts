---
layout: post
title: "Dynamic Library Structure"
date: 2021-04-30
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Dynamic Library Structure

Hi.
My name is Edward, and welcome to today's episode of the Pytorch dev podcast.
Today, I wanna talk a little bit about someone's or perhaps saw anyone who is a software architect's favorite subject, the library structure in Pytorch.
Now what do I mean by the library structure in Church.
Isn't Piper just one library that everyone uses? Well, that's true in one sense in that, you know, we distribute a single pie torch wheel that people use and think of as one unit.
But internally in our library, pie torch is actually split into multiple separate dynamic libraries, at least an open source, but this is also true inside our internal build system.
It's split into multiple different libraries, no naming, ranging from c ten, a ten core, a ten, torch, torch, Python.
And, you know, each of these libraries is, you know, a proper unit of encapsulation.
and means that you can't, for example, will a nilly depend on something from George Python, from c ten.
If you're not very familiar with, you know, what people are using these libraries for, you might think that this is just a whole waste of time.
Right? Like, you try to write some code, you put it in some folder, and then you have to decide which folder you're gonna put it in, and then it turns out you put it in the wrong folder and you've got to, like, move some stuff around and make everything work out, it it can really feel like a waste of time for no good reason.
And some of the library structure in Pytorch is vestigial and, you know, really shouldn't be there and we should, you know, reconsider how it's actually set up.
But a lot of the libraries in Metroid exist for some good reasons.
And in today's podcast, I wanna explain what the reasons behind the library split in Polish r.
And hopefully, that'll help you also think about how to better structure your code so that you don't accidentally, you know, violate one of these obstruction boundaries.
So, principle one that I would say about dynamic library, you know, structuring in general, like just how you decide to set up libraries is that for any major dependency you might have, it's usually a good idea to give it a separate library.
So a good example of this is CUDA.
CUDA is a really honking big dependency.
Right? Like, you've gotta actually have NVIDIA's CUDA runtime libraries And then there's, you know, actually a whole bunch of code in Pytorch that only really makes sense when you're running on a system that has a GPU.
We offer CPU only builds a PyTorch, which don't have any CUDA bits for people who don't have GPUs.
And the point of this is that, you know, Many people don't want CUDA and so there should be a way to use PyTorch without having to actually drag an all of CUDA.
And you add Pytorch's one single giant library in the situation, that wouldn't work.
You'd, you know, have to always get in the CUDA dependency.
Well, you might say, hey, Edward, you know, isn't the normal thing in open source to give you a bunch of configure flags and you just ask for which features you want.
And the answer is yes.
That's true.
Like, if you've ever built Python for from source, for example, there's a whole bunch of flags you can toggle on and off.
But if you're actually working in, say, a Linux distribution or you're working inside FB code, Typically, it's frowned upon to recompile the same piece of software multiple times with different flag settings because well, you know, how are you gonna distinguish between all these different versions? So when you're in a situation where you can only ever build some piece of code once Well, you had better not, you know, find you'd better find some other way besides if defying to split things out.
And so in PyTorch, we have a a ten CPU library that has all of our CPU kernels.
and we have an eight ten CUDA library that contains all of our CUDA kernels.
And so if you're say in buck and you wanna depend on a library, but you don't want any of the CUDA functionality.
There is actually a dependency you can depend on, the CPU only dependency that will prevent you from bringing in all your CUDA code.
So if you look at another really important library torch python, this one is also split off from live torch.
And why is it split off? Well, because Lipitor's Python has a dependency on the c Python API.
And there's plenty of situations when you are, you know, doing a c plus plus only application, you don't actually want to have the dependency on Python.
So that's principle one.
Whenever there is a major dependency, there is probably a library split looking nearby.
Principal two is sort of related, but more of an internal concern, which is that you want to split so that you can use what you need.
So what do I mean by that? Well, in many situations, binary sizes at a premium, and you don't want to actually ship code that you don't actually use.
So you know, honestly, principle one is sort of the extreme version of this where the, you know, thing you're not using is a giant, you know, honking blob of code that is from someone else.
But, you know, Pytorch is also big in and of itself, and we don't want to necessarily use code in Pytorch if, you know, we don't need it.
We don't wanna actually put things in if you don't need it.
And so similarly, parts of PyTorch are split in this way so that we can actually dispute these things without all of the functionality in question.
So one good example of this in high touch is the split between a ten core and a ten.
Although, This plays a little historical because mobile is deciding to ship more and more stuff.
In the beginning of the project, there was only a very limited subset of functionality that needed to be shipped on mobile.
And so when, you know, we When we wanted to actually put Pytorch into production, we wanted to actually merge the Cafe two and Pytorch code bases we needed to find a way to, like, put in the code that we wanted on mobile in one place and all the code that, you know, wasn't relevant to mobile in some other place.
and that's why a ten is split into a ten core and a ten.
A ten cores the stuff that's relevant for mobile and a ten is everything else that, you know, you might not be so interested in.
I say the split is a little historical because as time has gone on and mobile has gotten more and more features, it turns out that a ten does provide a bunch of stuff that mobile wants.
But in the beginning, it didn't.
And a ten core is this sort of minimal version that, you know, is generally applicable and takes up less binary space.
than all of a ten.
Another good example of this is the torch in a ten split.
So a ten is short for a tensor library and originally it was received of as just a way to do PyTorch code.
Like, you know, you wanna do an add, okay, a ten will tell you how to add two tenses together.
Whereas torch is the lag the library that actually gives you all of the sort of neural network functionality.
So it knows how to do automatic differentiation.
It knows about n n modules, all that good stuff.
And so once again, if you're in a situation where you don't act care about doing a d.
You don't care about doing your own networks.
You just need a way to do some tensor commutations.
Well, the split between a ten and torch means that you can just use a ten in that situation.
So that's principle two, which is split on what you need.
a more, you know, sort of internal version of split on dependencies.
And principle three, is kind of a cop out, but it's really important, which is we split our libraries for technical reasons.
That is to say, sometimes, there is no way to actually ship Pytorch unless we actually have things split in some particular way.
Let me explain one particular example.
So a very sort of right of passage for any new developer on Pytorch is writing a new function and forgetting to slap a torch underscore API macro on it.
You'll get a very obscure linker error saying, hey, you know, I have no idea what this symbol is even though, you know, like a compiled fine and the symbol is there, what the heck is going on.
So why does this macro exist in place.
Well, this macro exists because of something very interesting.
So I I I have to take a brief detour to explain.
So When we write dynamic libraries, we have to specify what symbols we actually expose as opposed to private symbols which aren't available.
to external users.
That kind of makes sense.
And if you're writing a, you know, standard Linux library, you usually just expose everything.
Like, you don't really care about very much hygiene in this case.
But on Windows, there's actually a problem, which is the Windows DLL format only allows for about sixty five thousand exported public symbols.
Now, sixty five thousand would be a lot of cookies to eat But as far as symbols go, it's nothing.
And a any, you know, self respecting project is gonna quickly hit this limit.
So on windows, because of this limitation, people tend to be a lot more careful about what actual symbols they put in their libraries.
So you have to actually say, you know, what symbols you want.
And if you, you know, if there's a symbol that you don't want, you just don't make it public.
So on Windows, we have hidden visibility by default, and you must explicitly export a symbol you want to.
and guess what macro does that? Well, that's the torch API macro.
Okay.
That's cool.
But what does that mean? Well, Remember, the symbol limit still applies.
Just using the torch API macro doesn't mean that, you know, you're not continuously adding more and more symbols.
And it turns out that the consolidated PyTorch a ten and a ten CUDA libraries goes over the Windows symbol limit if you put them together.
So no, we cannot ship Pytorch unless these libraries are separate.
so that we are under the public symbol limit.
Another example of a technical reason requiring us to actually keep the library split is for mobile.
So mobile mobile started off, you know, just having a small dependency on a ten.
But eventually, they actually needed operators.
But there's a problem.
Right? Which is that A10 has a ton of operators and mobile doesn't really want most of them.
Like, there's only a few operators that are actually used by models in practice, and they'd much rather prefer to only ship those operators.
So mobile has some very complicated system for recompiling Pytorch so that, you know, only the operators they care about are compiled for any given library.
Okay.
That's cool.
Do why what do I recompile in this case? Well, library split comes to Rescue.
Because we have all of our CPU kernels in a separate library eight and CPU, that's the only library that get needs to get recompiled on per app basis for mobile.
a ten itself which just contains, you know, common code that's used everywhere doesn't need to get recompiled in this situation.
So, you know, having library split in this way made it easier for mobile to do selective build.
And if you ever propose merging these things together, Well, you you'd better have an answer for what you're gonna do on the mobile side.
So what are the principles behind PyTorch's library's foot? well, you know, whenever there's a major dependency, that usually means there's gonna be a library split.
We split because that way lets us, you know, let people use code that, you know, use what you need.
You know, we don't we don't go to the, you know, extreme with this because It's very hard to deal with lots and lots of Anybody libraries.
But, like, for major partitions of functionality, there will be a library split usually in that situation.
And finally, there are a bunch of weird ass technical reasons like, you know, windows and mobile that also require us to split things in this way.
Okay.
So that's why we have so many libraries in PyTorch.
Some of the libraries probably can get merged together like a ten core and a ten probably can be merged together.
c ten probably could be moved into a ten except there's this funny business with our AMD Rockom support where simplification works differently.
in one case or another, yeah, it's complicated.
There's a lot of things that sort of accretive over time.
But, you know, usually, if you're running into a library problem, the best fix is not to actually, like, rage against the live restriction pie chart, just just to do a few simple things to, you know, sort of unblock yourself.
So what are those things? So one thing you can always do is sometimes some code is put in the wrong place and so you just need to put the code in the right place.
Right? Just move a file around yeah, I know it's annoying.
You can always put a little stub in the old location so that you don't have to update all the includes.
But, you know, oftentimes, just moving a file to the appropriate place because, you know, whoever put it there originally didn't think too hard about it.
That often will solve a problem you have.
Of course, Sometimes you do need to break layering.
Right? Like, sometimes you need to be able to call into some code in, say, torch.
when you're inside c ten.
And there are no amount of moving files around that will save you.
And so there's another trick that's, you know, sort of used very commonly in the code base, namely making a virtual interface and that you can call into the, you know, higher level library layer from a lower level library layer.
So one really good example of this is device guard.
Device guard works by having a device guard interface for every implementation of the device card.
And so if you're in a situation where you don't necessarily know if you have access direct access to the library in question, you can use device guard and it will do a virtual jump to the ash implantation, which might be CUDA to actually get the functionality that you want.
Of course, if you're actually in the CUDA library, you don't have to do this virtual jump.
And so there's actually a specialized version of device guard called CudaGuard, which lets you do exactly this when you don't need to violate the layering.
So that's all I wanted to say about library structure today.
Thanks for listening.
See you next time.
.
