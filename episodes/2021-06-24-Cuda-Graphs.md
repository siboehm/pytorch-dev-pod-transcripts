---
layout: post
title: "Cuda Graphs"
date: 2021-06-24
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Cuda Graphs

Hello, everyone, and welcome to the Pyturbine Podcast.
Today, I want to talk about CUDA graphs, an Nvidia mechanism for reducing kernel wash overhead and, you know, sort of putting all your CUDA kernels together into one mega kernel that you can run really fast.
So why does CUDA graph exist? Right? So To understand this question, we have to think a little bit about how the CUDA programming model works.
So the way the CUDA programming model works and see my previous podcast about enough CUDA to be dangerous.
The way the CUDA program model works is we have a bunch of kernels that the CUDA, you know, CPU knows how to run and you run your host code, regular old CPU code code, and you figure out what kernels you wanna run and you queue them on a stream and, you know, like whenever the CUDA driver gets these kernel launches, it actually goes ahead and runs them on your GPU.
And so if your data is really big and, you know, like, it takes a long time to run various things in the GPU, after a short launch latency, the latency that it takes to get to the first CUDA launch, then you will basically just queue a bunch of kernels to be run on the stream and, you know, CUDA will just go ahead and try to, you know, run them as fast as possible.
when the previous work gets done.
But sometimes, your code is too small and it runs too fast store, maybe NVIDIA's graphics cards are way too fast.
And you've got a problem which is you just can't keep up with the GPU, you can't feed it enough to keep it utilized.
And, you know, when you're in this regime where your tenses are really small and you have a lot of antibody current launches, current launch overhead actually can be pretty killer.
And so, crudeographs are a solution for this problem.
When a crudeograph lets you do lets you take a whole bunch of kernel launches and bundle them up into one giant mega kernel launch so you don't have to deal with the kernel launch overhead and, you know, you can you've gone a bit of all that overhead, you've gone a bit of the overhead of running the host codes, so your CPU overhead is also lower.
Your CPU utilization is also lower.
And then you can just go ahead and, you know, run this over and over again.
Okay.
So that's the concept behind CUDA graphs.
But if I told you, hey, I need you to go implement CUDA graphs for me.
you might think about it a bit, and then you might realize, actually, this is not so easy to do.
Right? Like, so normally and, like, if you're say ML Kim commute at Apple, you know, this is what you actually did.
Normally, what you would imagine is, hey, you know, I want some sort of graph representing the entirety of the computation that I wanna do, and then I'm gonna feed it to some sort of, you know, internal engine, etcetera.
And that's gonna, you know, go ahead and, you know, compile into one model kernel can go ahead and send Nvidia.
But there are known as such graph representation exists for CUDA.
Right? Like CUDA was designed from the very beginning as a streaming API.
And so what's actually going on, right, is, like, in PyTorch, we've got loads and loads of Kuna hurdles all over the place.
they they don't even necessarily have to be, you know, like, have a publicly visible name.
They can be in in an anonymous name space.
And they've got all of these, like, you know, parameters that you're calling them with, right, like all the tensors that they wanna operate on, various, you know, parameters that you're passing on the parameter buffer to the kernel, like, you know, whatever scaler you wanna multiply things by or anything like that.
Like, how the heck would you actually assemble a graph like this? And so crude graphs like, you know, many other wonderful technologies such as the jet torchcraft tracer requires you to go and run your CUDA kernels first and record a CUDA graph that you actually then can run again in the future.
That being said, there is a API in Crediglass for explicitly building pseudographs and doing modifications to modifications to them after the fact, but that's not the preferred way of generating a pseudograph.
The preferred way of generating a pseudograph to actually run your code once, and then you actually get a bunch of Crudoclonal launches.
And by the way, like when you do these Crudoclonal launches, you know, we're gonna record everything about how you launch them.
Right? So, like, what tensors you're passing to them what parameters you're passing to them, all of that we're going to just record as is.
So that means that it's totally hard coded.
Like, if you use some CUDA memory inside your region of CUDA calls, that memory is going to be the very same memory that a subsequent run of the cutograph is going to use.
Because remember, In contrast, it has no idea what the meaning of the parameters you're passing through the Kono kernels are.
Like, it's totally flexible.
You can you can pass anything you want.
You can pass any structs you want.
So CUDA has no way of actually just swapping out pointers if you wanted to like, you know, use different memory the next time you run it.
So when you're doing CUDA graphs, you have to like you know, make sure that you allocate your memory in a persistent way so that the next time you wanna run your code, you can reuse that memory.
for that.
So the model behind CUDA graphs, right, is that you you run your CUDA code with a special setting on the memory allocator so that, you know, it gets kept for later.
And then once you get done, you get this scooter graph.
And for whatever the input scooter tensors are, You have to go fill them in with whatever the new inputs you wanna run and that situation is.
And then you can say, okay, Nvidia, go run your crude graphs and bam bam bam, it'll go ahead and run the kernels exactly as they did previously.
Oh, yeah.
And one last thing, because you know, how exactly do CUDA graphs know what kernels to actually record? Well, actually, they're stream based.
So remember, the stream in CUDA is this queue that keeps track of all the operations and what ordering they need to run-in.
Right? So if you put things on the same stream, they're guaranteed to run-in the order they got put in the stream.
Of course, if you have multiple streams, then they can run-in any order.
And it's a little hard to use streams correctly because, like, it's very, like, fine grain form of parallelism and like, sometimes physically your GPU just can't do it, but it is a useful API.
And so CUDA graphs, when you record, you're not recording globally every CUDA launch, you're actually recording CUDA launches on specific streams.
And PyTorch is not that great at being very stream friendly.
Like so, you know, Pytorch by default runs on the default stream.
The default stream synchronizes with everything.
It's very easy to use You don't have to worry very much about it, but, like, you know, sometimes you want to have streams and then you have to actually write your code differently.
It's easy to get this wrong because if you forget to do it and someone runs your code on the default stream, chances are things are just gonna work out.
So, you know, temporarily, who is the Nvidia guy who's been working a lot on crude to graph support in Pytorch.
He's also had to fix a bunch of stream bugs especially in our autograd engine to make everything all work out.
So that's basic most of what you needed to know about CUDA graphs.
Right? So they they are a way of running a bunch of CUDA kernels altogether at once and they hard code all the parameters so that just leads to some, you know, UX problems that you have to be aware of if you wanna use them.
I want to recap something that I talked about in the random number generator's podcast, which was about the Philoc random number generator because this has a very interesting interaction with CUDA graphs.
This is kind of bonus material.
So, like, I've already said the most important thing about CUDA graphs.
but this is I think this is interesting and I wanna talk about it a bit.
So I said that, you know, everything gets hard coded.
And in particular, the random number of state gets hard coded when you run your CUDA graphs.
Okay? Think about it.
Right? So what I said in the RNG podcast is that the CUDA RNG state actually lives on CPU.
It doesn't live on CUDA.
It lives on CPU.
And you just you pass the seed and the offset directly in the kernel parameters.
And then on the CUDA kernel, it actually sets up the Phalox state.
and then does sampling on it.
And it's pretty cool and it's very nice and it's a complete disaster for coup degrasse because what that means you are actually gonna get the same random numbers every single time you run your credit graphs.
Okay.
Maybe that's okay.
But like usually that's not okay.
And you really do want different random numbers every time.
So how do the heck do you solve a problem like this? So clearly, you need some way of actually feeding in what part of the sequence or the seed or something like that.
inside CUDA memory because, well, you know, you're gonna totally hard code the you're gonna hardcore the parameters.
Right? So it can't be anything passed in the parameters.
Well, there there's only two ways you can pass information to Accredo kernel, either by the parameters, or by memory on the CUDA device.
So if it can't be in the parameters, well it has to be on the device.
But then how exactly can you get it to the device like Do I have to you know, when I launch my kernel, first do a hosted device copy of their r and g state.
to CUDA memory and then run the kernel that way.
That doesn't sound so great.
To be fair, it wouldn't be that bad because remember it's all async.
and so you can trigger this well, as long as the host memory is pinned, which is not too hard to range, you could just trigger it asynchronously and then, like, have the transfer happen whenever, like, CUDA gets around to doing it.
But there's a better way to do it.
And the better way to do this is to pass in a pointer to a little bit of CUDA memory It doesn't say what the seed or the offset should be, but instead is an offset correction.
So what's the idea? So we're gonna put on a restriction.
The restriction is that if you want to use CUDA graphs with RNGs RNGs, you have to reuse the same seat.
because the seed we're sending up with the parameters.
So the seed is hard coded.
We can't do anything with it.
But what you just wanna do is, right, when I do subsequent calls to the crude graph, all I want is to, you know, advance the random number stream.
However, far, you know, I had advanced, you know, via my previous consumption as well.
Right? So there's only this, you know, extra bit of an remuneration just the offset that I want in this situation.
So what I can do is So when I'm running normal Pyters code and there's no CUDA graphs involved, I'll send a little bit inside the parameters field saying, hey, this is non capturing.
You can just do use the seat in the office directly and you don't have to do anything about it.
But let's say that I am in capturing mode, then I'll do a different bit and I'll send a pointer to the memory that is the offset that I want to do and say, hey, hey, when you compute the RNG state, use the seed, use the offset, but also use this extra offset re read out from memory to like do the adjustment.
Now the very winning the adjustment is zero.
Right? Because, like, whatever the seed and the offset were at the time I was recording is the correct one.
But then later when I wanna rerun the CUDA graph, all I need to do is do a, you know, a host to device setting of that little bit of offset to be whatever the current state of the RNG is.
And now I can run my crudeograph and the crudeograph is gonna read out the, you know, the offset from this memory and now offset the random numbers exactly how I need them to be.
And there's one last thing I need to do this, right, which is I need to know many random numbers my pseudograph consumes, but that's not too hard to figure out.
You just record what the r n g state was at the beginning and what the r n g state was at the end.
This was not obvious to us at the very beginning.
And, you know, Emily, Natalia, and I, like, spent a while thinking about how to actually solve this.
But I think this solution is very elegant.
And it's just, you know, once again, it comes out of having to solve the problem of while CudaGraft's hard code, everything in the parameters.
Actually, in an old version, apparently someone was actually going into the coup de graff post facto and editing all of the RNG parameters update them to a new thing.
This was terrible.
It was a bad idea and, like, needed to solve this problem.
Okay.
So that's the end of the fun technical digression.
So kudos.
So, like, how can you actually use them in practice? So we're working on landing the last PRs that actually give a nice user API But there is something, you know, that is very important about Crude Grass.
Right? Which is if you wanna deploy them, you wanna use them in a production setting, you need to be able to run your code you know, initially to actually get the crude graph in question.
And so this is why, like, things like torch deploy are actually very important for crude graphs.
Right? Because, like, if you wanna use crude graphs to, like, do, say, GPU inference because that's a situation where overhead matters a lot.
you still need to bootstrap the CUDA graph at the very beginning, and then, you know, then you can run it.
And, you know, if you You don't if you can run Python code in your environment and that's what torch deploy is all about, then you can just run the slow Python code to get the crude graph but then pass it off to some c plus plus, you know, engine that just repeatedly runs the crude graph in the future.
Right? And that that'll be really good and, you know, you you you use the Python for the slow initialization, and then everything else doesn't even need to touch Python at all.
And that's like I think one of the main draws of CUDA graphs.
Alright.
That's everything I wanted to say for today.
Talk to you next time.
.
