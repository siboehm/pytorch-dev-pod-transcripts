---
layout: post
title: "Just Enough Cuda To Be Dangerous"
date: 2021-05-05
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Just Enough Cuda To Be Dangerous

Hi.
My name is Edward, and welcome to today's edition of the Pytorch developer podcast.
Today, I won to do a very whirlwind intro to CUDA programming.
Now disclaimer, I am by no means a CUDA programming expert.
I've ridden a CUDA Colonel or two in my time, but most of the time, I defer to such experts as, say, Natalia Gimmele Schein.
to actually do the heavy CUDA lifting.
But having worked on PyTorch a while, I have picked up a thing or two about CUDA.
And so today's episode, I just wanna, like, talk about really, really, really fast.
You know, here's just a big pile of stuff that is important to know about CUDA programming about programming GPUs in general just enough so that you can be a little dangerous even if you're not like actually riding CUDA Crohn's.
Because It's really helpful to know a little bit about the programming model.
What happens on GPU? Because while, you know, PyTorch is a GPU accelerated deep learning library, And so if you add some functionality to the PyTorch, we expect you to be able to also run it with GPU acceleration.
Alright.
So we're to get started.
Part one, what is CUDA? So to answer what is CUDA, we actually have to answer a different question first, which is what is a GPU? So the GPU is a piece of hardware in your computer that sort of made the deep learning revolution possible and its name is short for graphics processing unit because historically, that's what we actually use them for.
We use them to actually, you know, render graphics scenes on your computers if you're playing a video game or, you know, doing some sort of photo photo or video application.
And it just turns out that the types of things GPUs are good at doing are also good at doing deep learning models.
Why is that the case? Well, the way a GPU works is that instead of having so remember when we talked about vectorization and I said a CPU, you feed it a bunch of instructions and it does the instructions one by one, and you know basically that's it.
And you can like put a bunch of cores in your CPU.
And if you have a really beefy machine, maybe you have thirty two or sixty four cores, But, you know, there's only so many cores you actually put in your CPU, and that's basically it.
Like, that's the level of parallelism you're gonna get.
You have to, you know, spawn threads and, you know, use them to actually make your CPUs go.
Well, a GPU has tons and tons of really, really simple cores And the way they operate is they just say, okay.
Well, I'm gonna run the same computation on every core, so I don't have to worry about, you know, all the cores doing different things.
And I'm just gonna have so many cores doing the same operations that if I have a ton of data like say in a image or in a deep learning tensor, Because I have such massive parallelism, I can actually just, you know, do things very quickly.
Because even if there's a million things to do, well, you know, I have a lot of cores and so they can make quick work of it.
So the basic idea behind GPUs is instead of having, you know, these big biffy CPUs but not that many of them.
We have these many many cores and we massively paralyze our algorithms, and that's how we're actually gonna do things really fast.
And so CUDA is the programming language slash compiler stack slash software ecosystem that Nvidia developed for programming their GPUs for, you know, sort of general purpose programming.
Because back in the day, like, when you had a GPU, you used it to do graphics processing.
So you'd write shavers, you'd write, you know, those sorts of things, and no one was really thinking about doing, you know, actual mathematical general purpose computation except for, you know, weird branch of researchers who are looking into so called g p GPU's general purpose GPUs.
And they would, like, go through lots of tricks to try to, you know you know, get the shaver to do exactly just the thing that they wanted them to do for whatever computation they wanna do.
And Nvidia built this software stack called CUDA, and so we can use CUDA to do general purpose programming on GPUs.
And in PyTorch, What we do this for is so that we can do deep learning neural network computations on them.
So what is the CUDA programming model? So, you know, the GPU is not your CPU.
Right? Like, on the CPU, if you wanna do some stuff, you just send some instructions to the processor and, you know, it does just just does the stuff.
You don't have to think about it.
Right? That's normal operating.
But your GPU is typically living on a separate, you know, device in your CPU and like it's got its own memory and it's, you know, not like anything at all like your CPU So there's actually a little bit of difference when you wanna program a GPU in this situation.
And so the sort of very, very short version of, like, what you should think of as a CUDA programming model is there is CUDA memory.
That memory is memory that lives on the GPU.
That's the memory that, you know, programs on the GPU can actually run.
If you wanna compute some data on the GPU, you have to first move it to the GPU so that it's accessible.
Then you can write various kernels, and these kernels are, you know, sort of written because CUDA is a programming language built on top of c plus plus.
They're written in c plus plus.
but they're different than normal c plus plus because, you know, unlike a regular CPU where, you know, you have a single processor and you just feed it instructions, these programs need to work as this running on, you know, all the little antibody processors that are on your GPU.
And so these special kernels, you you you wanna go right a them in a subset of c plus plus that, you know, your CUDA compiler actually understands.
And in general, like, when you write a program or, like, in PyTorch, there are gonna be, like, you know, dozens or really hundreds of CUDA kernels each for some per particular task that you wanna do.
I'm not gonna talk about how you actually write a CUDA kernel today, but say you have a bunch of these kernels.
What you need to do is after you put the data on the GPU, you need to ask the CUDA driver hey, can you please run this kernel? And the CUDA kernel, well, CUDA driver will go ahead and say, okay, I'm gonna go tell that GPU that the actual device to go ahead and run this computation to do the thing that I wanted to do.
And here is sort of one of the most important things about the CUDA programming model.
The the most important thing If you've never been included before and there's one takeaway I want you to get from this podcast, it is this process is asynchronous.
I'll repeat again.
This process is asynchronous.
So you tell the driver, hey, please do this commutation.
The driver is like, okay, I'm gonna go do this computation.
And the kernel call you made is going to immediately return.
Even though the GPU is off, you know, chunkering away on the data that you asked it to process.
This is a good thing because it means that your CPU host program that's responsible for figuring out what kernel calls do, can run ahead while the GPU computation is happening and figure out what the next thing you want it to do is and so you can say, hey, after you're done doing this previous computation, please do this next computation and you can queue it ahead and the GPU can be ready to go right when the previous commutation finishes.
By the way, how does it know that it wants that kernel to run after the previous kernel you ran? Because if it's asynchronous, couldn't these just, like, run-in any order, couldn't it just start running it when you ask it for it? Well, there's this thing called streams.
streams implies sequential execution.
So you put CUDA kernels on streams and every kernel on a stream is guaranteed to finish before the next kernel on that stream happens.
Normally, when people just write GPU accelerated programs, there's just one stream.
It's the default stream.
Everything goes on that.
Everything is sequentialized.
But if you're doing like fancy tricks, you might have multiple streams.
And one of the things Pytorch needs to do is although most people don't use streams, we do want it to be possible to use streams with our software.
So we have to write all of our code in a stream generic way.
One last thing that's useful to know about the CUDA programming model is it has a notion of a current device.
So, you know, when you do a kernel launch, well, you might have multiple GPUs in your machine.
Right? And each of these GPUs has its own memory.
And so you can't just say, oh, well, you know, GPU two, please operate on the memory and GPU zero.
Technically, this will work.
if you have, you know, device to device transfer, but it'll be kinda slow.
Right? So most of the time, we don't allow it.
And, you know, you have to actually make sure the memory is in the same place.
And so the current CUDA device, which is a CUDA concept, is something that you have to say, okay, I am now setting my current device to be GPU two so that all of my kernels actually operate on GPU too.
Because the kernels don't actually take in what device they wanna run on explicitly.
Plytorch also has a notion of a current stream.
This is not a CUDA concept.
This is something that Pytorch built on top of CUDA.
And this is so that we don't have to also constantly say which stream we wanna run on.
CUDA kernels explicitly take which stream you want or zero for the default stream.
Okay.
So that's the basics of the CUDA programming model.
So what are the implications of this model when we are doing Pyturgical programming.
So remember I said the most important thing about CUDA programming is it is asynchronous.
So what happens if something bad happens in your CUDA kernel? Because bad things can happen in your CUDA kernel.
They're basically c plus plus.
Right? You can do an out of bounds point or do you reference You can have an assert failure.
You can, you know, trigger a a, you know, compiler bug.
Any any lots of things go wrong.
Right? So what happens when something goes wrong? Well, first off, when you launch the kernel that actually is going to do something bad, it's not going to raise an error.
Right? It's just gonna return and say, hey, everything's okay.
But that's not actually necessarily the case.
Right? Because some later in point in time when the drivers finally got in ahead to getting to figure out, hey, you know, there's something wrong because I've just run this kernel and get it.
you might be somewhere way else later in your CPU HOSI program, at which point the CUDA driver you'll be doing some random call into the CUDA Kuda Kuda API, like, trying to Malek something or, like, trying to launch a different kernel and say, oh, no.
No.
No.
No.
Something that has happened.
An internal assert failed.
I don't know.
And well, crap, because, you know, you've got this code and it has nothing to do with the error that just got raised because the error was actually caused by some kernel launch, you know, miles and miles away in your code.
So this is like the most, like, you know, anyone who, like, just sort of, like, signs up for PyTorch and doesn't know any CUDA and, like, has to debug a GPU problem.
This is probably the first thing you're gonna run And you're like, oh my god, what the heck is going on? And the answer is remember, let's say you think, you're getting the results way later after you actually kill the kernel.
What can you do in this situation? Well, there's a bunch of things you can do, but the easiest and, you know, simplest way to, like, solve a problem like this is to use this environment variable called CUDA launch blocking, which says, hey, you know, wait until the previous kernels have all finished.
before actually executing my kernel.
And in this case, because we're waiting, we can actually make sure that we you know, have gotten all the errors before we move on and try to do the next operation.
So that will cause the errors to move to the right place.
Your programs will run really slow because remember asynchronous execution is a good thing.
It lets us make sure we keep the pipeline, our GPU computation going, whereas, you know, with blocking, you're gonna wait.
And then the GPU's gonna idle while your, you know, very slow CPU host tries to figure out what the next thing to execute is.
until you get to the next thing and then it's gonna run again.
Right? So your utilization is gonna be crap, but at least you know where the errors are going.
Let's talk about this asynchronous thing again.
Right? So we said that, you know, CUDA programming has to, you know, run ahead so that, you know, we can make up for costs of, you know, launching overhead and, you know, waiting for CPU to figure out what the things to do.
Well, there's another consequence to this.
Right? Which is anytime you ask for some memory that's in CUDA in your GPU and you wanna actually like, look at it on the CPU, like, you're gonna say, oh, is it a two? Is it a three? Can I do something with this? You have to wait.
Right? You had to wait for all of the asynchronously queued kernels to finish executing so that you can actually see what the data in that memory is.
And then you have to copy it back to CPU and then you can actually go look at it.
So syncs are really, really, really expensive.
And whenever we write code in PyTorch, we really want to try to avoid doing synchronizations that are unnecessary.
And sometimes this is not so easy to do because there are a lot of innocuous sounding methods that can cause synchronizations.
For example, if you ask for torch dot non zero on a CUDA Tensor, that will cause the same.
Why does that cause a sync? Well, it causes a sync because non zero gives you a tensor whose size is the number of non zero entries in the original tensor.
you know what the non zero entries are? Well, you have to look at the data, sync.
Another example is dot item, which, you know, takes some element somewhere in a tensor and then gives you what its value is and you look at this and you're like, oh well, I got this thing from CUDA memory.
So that means I had to wait for all the computation to finish to get that thing from CUDA memory.
So try really really really hard not to do things.
sometimes this is impossible.
Right? Like, maybe you're doing some iterative algorithm and you're, like, you know, repeatedly running some kernel and waiting for some value to converge.
before you do thing, before you stop and go do something else, well, yeah, you're kinda out of luck.
Right? You're gonna have to actually sync when you do that.
But there's often some way to things up so that you don't need to do the sync or maybe there's like a different version.
Right? Like, there's a fast version that doesn't sync and then the slower version that does sync and you want to think about actually providing both of these things.
Speaking of asserts and sinks, Remember what I said about, you know, like your errors showing up in way random places.
Right? So In PyTorch, we actually have this philosophy, which is that we are willing to pay a performance cost in our CUDA kernels so that we get good error reporting.
Let me give you an example of this.
Say you're writing some sort of embedding and so what is it embedding? It's just a glorified hash table lookup.
Right? So you you got some index you wanna go look at the element at that index.
Right? What if the index is out of balance? Well, we could say, oh, you know, we really care about performance.
we wanna we don't wanna bounce check.
We're just gonna do the the reference.
And if there's, you know, if it's out of bounds, Well, too bad for the user.
Right? Like, you asked for it.
It's up to you to make sure things are inbounds.
We do not make this assumption.
We will bounce check these accesses.
For one, it's not that expensive to do because, you know, your this massively powerful CUDA, you know, GPU device and, you know, you're gonna be spending lots of time usually being memory bound.
So, like, you know, extra computation usually isn't that expensive.
But two is that if you do a, you know, invalid memory access, you're just gonna get an invalid memory access and you have no idea what could have caused this problem.
If you do a bounce check and you do an assert, you will get that assert when things fail later.
And so you can, for example, grab the cartridge code base and they'll tell you, hey, this is what caused the assert.
And then you can have some clue, oh, it was this operator without having to run CUDA non blocking.
So I have told you a little bit about what GPUs are.
I've told you about the CUDA program model, and then I started harping over and over about syncs async, all that stuff.
Right? Because really the asynchronous nature of CUDA is what really really trips people up.
In fact, like, even in advanced usages, like this these streams or you have multiple streams, like, making sure all of the, you know, synchronizations between streams happen correctly, and happen correctly with, say, our CUDA cashing allocator.
Oh, yeah.
We have a cashing allocator because CUDA alloc is really slow.
So we get a bunch of memory from CUDA and then we, you know, reuse it for our own stuff.
But making sure this all gets synced up so that like async stuff doesn't messes up, yeah, that's like probably the hardest thing about working in CUDA.
So if you can remember, async is cool, but it is very complicated.
and make sure to remember that when you're working on CUDA, you're a long way even if you don't know anything about how to write CUDA algorithms like me.
Alright.
So that's all I wanted to say today.
Thanks for listening.
you next time.
.
