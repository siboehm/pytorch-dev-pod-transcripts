---
layout: post
title: "Vectorization"
date: 2021-05-03
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Vectorization

Hello, and welcome to the Pytorch dev podcast.
My name is Edward, and today I want to talk about vectorization.
Vactorization is a very important component of any self respecting deep learning or really any numeric computing library that lives on CPU but sometimes it has a bit of a reputation for being this, like, very mysterious, very magical thing.
You know, numerical codes go into compiler, vectorized instructions come out, and you know, you're not really meant to know how exactly the sausage is made.
Well, actually, you know, vectorization isn't that magic.
And today, I wanna talk a little bit about how we make use of vector instructions in PyTorch.
on what vectorization is, and some of the sort of tips and pitfalls associated with vectorization in the code base.
So what is vectorization? Well, imagine that you're doing some computation on your CPU.
normally, the way a CPU works and what you learned in your architecture class is you have a bunch of instructions You feed the instructions into the CPU, and the CPU goes ahead and does the things that you ask it to do.
So for example, if you, you know, wanna do an ad, you tell the CPU, hey, I wanna add this number and this number together from these two registers, and the CPU will go ahead and do that for that single instruction.
Now as you might imagine when we're doing numeric computing, we don't have just one number.
We have a lot of numbers, and we wanna do the same thing to all of these numbers.
and that's where vector instructions come in.
Vector instructions are a form of what we call SIMD parallelism, that's SIMD, single instruction, multiple data, or instead of skipping your CPU and instruction to do an operation on a single piece of data, you can give your CPU an expression to work on multiple pieces of data.
That's why they're called it's called vectorization because you're working on a vector of numbers rather than one number.
So when you wanna write some vectorized code, you have a bunch of these vector registers, which are larger registers than you normally be able to use to do various computations.
The idea being you'd like fit in multiple numbers into these registers, and then you have a whole pile of new instructions to do things like add but not just add one number, but add all of the numbers in your vector registers.
And the vector instructions are actually pretty simple And so if you wanted to, you know, go and learn how to, like, you know, write some vectorized code by hand, all you'd have to do is really pull up the intel manual or, you know, whatever, you know, manual for whatever processor you wanted to do.
And, like, just look and find which instructions you wanted to do.
Or you could use a library like sleep which already provides pre vectorized instructions for you.
Or you could even just, you know, write some code and hope that your compiler's auto vectorizer handles it for you.
You just you need to pass to fly like MAVX, and it will try its best to vectorize your code for you.
So on Intel CPUs, which are the CPUs that most people are using, the vector instructions are called AVX.
Stanford advanced vector in extension.
And there's a bunch of different versions of AVX, basically, because over the years Intel was like, you know, we only really wanna do vector operations on two pieces of data.
So here have an extension that does that.
Actually, that was called SSC.
And then over time, they gave more instructions, more bigger vector registers, and more and more features.
And so as time went on, you know, they released AVX then AVX two, then AVX five twelve.
And so just, you know, over time, there's more and more functionality But remember, and this is gonna be very important later in this podcast that, you know, you need a CPU that actually has the silicon for doing whatever it is you wanna do.
So if you got like a, you know, CPU from like two thousand fifteen, chances are it doesn't actually have AVX five twelve.
It only has AVX two.
You can actually find out what vector extensions were supported by your CPU on Linux by cutting out the contents of proxy CPU info.
that's a magic file that the Linux kernel provides that tells you all about your CPUs and tell you the model and it also tell you all the extensions that it supports.
And then you can look and see you know, which AVX is on there.
Okay.
So AVX is a bunch of vector instructions.
I'm not really here to teach you like how to write AVX code.
I actually have no idea how to write AVX code by hand.
Instead, in PyTorch, we have a bunch of extractions to make it easy for us to manually vectorize our code.
Because often we don't really trust the compiler to do a good job in vectorization.
So we just wanna you know, actually tell, hey, here are the exact instructions I want you to use so that there's no possibility for the compiler to mess it up.
And then the set of header files, which help us do vectorization in Pytorch are called the VEC aptly named VEC headers.
And so currently in Pytorch, we don't have support for AVX five twelve.
We just have support for AVX two, AKA, AVX two fifty six.
so called because the registers are two hundred fifty six bps wide.
And so we have a class called VEC two fifty six.
which just represents a bunch of vector data stored in the vector registers and then has a bunch of operations like add, sub, you know, sign and so forth for doing vector operations on this vector piece of data.
So if you wanna write some vectorized code, chances are you know, you might just be able to, like, get Vect two fifty six and then get your data into Vect two fifty six.
And we actually have a bunch of wrapper functions like CPU kernel, which help, you know, handle all the fiddly, you know, edge conditions.
Because remember, vector instructions always work on four pieces of data.
So what if you've got seven pieces of data? Well, you have to do the vectorized instruction on the four, but then you need a manual loop to finish the last three.
So you, like, get your vectorized thing and then you just tell, you know, exactly what vector instructions you wanna do is by just calling these methods on deck two fifty six.
And if you wanna, like, actually implement some new and interesting functionality using the raw intrinsic the intelligence being various special functions your compiler provides that lets you just directly call various vector instructions.
You can do that too and typically you just go into the factor fifty six class and you write in exactly what instructions you want it to use in this situation.
So it's a pretty fun exercise to, you know, add vectorization support for something.
And, like, if you're sort of in the mood for just, like, you know, cracking open the intel manual and like reading some papers and trying to figure out how to vectorize something.
You know, a a pretty fun task is, you know, hey, I need to do something fast And right now, we only have these crappy, you know, single instruction implementation for it in PyTorch.
Maybe I can vectorize it.
Some things are easy to do.
Like, if you're just doing some point wise operation, you just need to figure out the right sequence of vector instructions to get the computation you wanna do.
Some things are harder to do.
I remember a Uman wizard way back in the day actually implemented a vectorized sword.
for PyTorch.
We never merged it because it was too complicated.
But, you know, like, that's the kind of thing.
Like, there's a ton of things you can accelerate using vector instructions.
And actually, they will run a lot faster on CPU if you do that.
So it's often worth doing it this way.
So that's it for what is vectorization and how people do vectorization in PyTorch.
And that's nearly it, but I wanna tell you a little bit more about some of the weird things that we do in the code base to actually make this all tick.
So remember this thing that I said.
Right? I said that not all CPUs support all vector instructions.
Depending on if your CPU is from two thousand ten or two thousand fifteen or from two thousand twenty twenty, you know, you're gonna have different support for vectorized instructions.
And no one really wants to, you know, try to run their Patrick's program and get a SIG legal instruction because, you know, you tried to feed the CPU some instruction it didn't understand.
And this is actually a bit of a problem for us because when you compile your code, That's when the compiler makes a decision to make use the various vector instructions that it has available.
But the compiler doesn't know where you're actually going to run the code later.
It's not like, you know, you're compiling some code and you're trying to test if, you know, you have a live XML on your system.
And if you do have it, then you compile the support for live XML.
Otherwise, you don't compile the support for it.
it's not like that because you actually have no idea where your end user is going to run your code.
And so, you know, you have no idea what vector instructions are gonna be available.
And so, you know, if you don't do anything special, you really can only ship your software for the lowest common denominator of CPU you want to support.
And typically, that's just, you know, no vector instructions at all.
Because you know, old CPUs have been around for a really long time.
So the way we work around this problem is, you know, we just say, okay, fine.
Some CPUs support factor instructions.
Some don't.
So let's just compile our instructions multiple times for each level of CPU support we want to support, and then just, you know, query the CPU processor at runtime, and use that to pick the particular comp compiled version of our code that actually does the vector instructions.
So we have a system that does this.
It's called dispatch stub.
Dispatch stub sounds very complicated, and in fact, you can also use it to dispatch to CUDA versus CPU.
But really, it has one goal in life and its goal in life is to let you get to the appropriately vectorized version of your code depending on what CPU capability you have.
So there's a bunch of macros and if you like sort of carboculture code, you can, you know, usually figure out how to make this work.
But the basic concept is in the native slash CPU folder, Any file you put in there will get compiled multiple times once per vectorization level that Pytorch supports.
and then each of these compilation units will register its kernel to dispatch stub saying, hey, I'm the AVX two fifty six version.
hey, I'm the AVX version and hey, I'm the non vectorized version.
And then dispatch stuff, we'll just, you know, query what CPU capabilities you have and then dispatch to the correct one.
And there's a bunch of sort of magic that has to happen to actually make this all work out.
For example, when you actually compile this code multiple times, you have to be really, really careful not to accidentally compile any other code that you don't actually want.
And this is important because when you compile c plus plus, normally, you would imagine you just compile the functions that you fine in your c plus plus file, but that's not entirely true.
When you do, so, example, template specializations, c plus plus will blot out another bunch of code and then sort of rely on the linker to deep duplicate this code later.
And so if you happen to blot out some code that in fact uses some vector instructions, and then that copy of the code overrides the regular version of the code that you compiled with no vector instructions because remember we don't want to we don't wanna assume that everyone supports vector instructions, then you can end up with normal code like vector resize using AVX two instructions and then your binary packages will be very unhappy because they'll, like, package the binaries.
It'll work all fine because all of our test machines are AVX two.
And then, like, some user is gonna report to us that hey, when I import torch, I get a single legal instruction.
What's up with that? Actually, we do have a test for this now in CI, so you don't have to worry about silently break this.
There's two more things I wanna say.
One is that if you wanna, you know, sort of If you've got a very featureful CPU, you can actually manually change what vector instruction you wanna do.
There's an environment variable that lets you do this escapes my mind at the moment, but you can look it up at it's it's got capability and its name in all caps.
And that, you can just use it to, you know, switch between versions and it's actually a pretty nice way to see like how much extra benefit you're getting at a higher level of vectorization.
One last thing.
So very very recently okay.
Not that recently at this point, but fairly recently Intel's ability support for the new AVX five twelve extension.
And so we've sort of been using it on and off, but we actually don't support it in the library proper.
And the reason we don't support it is because of this funny thing that happens to Intel CPUs when you start running AVX cycle instructions.
They down clock somehow, for some reason, when they design the CPUs, they, like, put too much silicon on it.
And if you, like, actually use the AVX five twelve silicon, it overheats the chips.
So they can't actually use all of the chips at this point in time.
So they down hook the processor to make sure their heat output isn't too big And that means that if you are switching in and out of AVX five twelve instructions and regular instructions, the down parking will actually kill your overall performance.
So we've kinda been, like, kinda low to actually add support for AVX five twelve.
But there's some very enthusiastic open source contributors who have been trying to add support for this at the framework level So go them, they're working on it.
If you're really interested, check out their PR, which I'm gonna post as a link in the rest of this podcast.
So that's everything I have to say about vectorization.
Vectorization, it's not magic.
Well, okay, when we recompile your code multiple times, that's maybe a little magic.
Hopefully, this explains some reasons why you have to put some code in CPU, some code in not in CPU.
Some of it is vectorized.
Some of it isn't.
And hopefully, it also tells you why you can't just, you know, use random templates inside the CPU folder because of symbol problems.
So that's all for today.
See you all next time.
.
