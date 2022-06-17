---
layout: post
title: "The Road To Structured Kernels"
date: 2021-05-10
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# The Road To Structured Kernels

Hi.
My name is Edward, and welcome to today's edition of the Pyturgical Dev podcast.
Today, I want to talk a little bit about structured kernels and meta tensors.
a project that I've been working on for the better part of a year, maybe more than that at this point.
Structured colonals are basically a new way of writing kernels in Pytorch where you can instead of writing, you know, a kernel from whole cloth that does all of the computation, all of the determining whether or not the inputs are right, and all of the output, shape, size, computation, for example, it allows you to factor your kernels into a structured form where you write a meta function, which says, you know, what the input checks need to be and what the output sizes are gonna be.
And then a natural implementation function which you can then do a separate implementation for CPU and CUDA.
And they reuse the meta function to do all the you know, shape checking, but then the actual implementation bits can be different in both cases.
And then meta tensors are a sort of easy extension on top of this.
which is that, well, once you have this meta function that all it does is check the input d types and figures out what the output shape needs to be, you can actually then do a third sensor type, not CPU or CUDA, but meta, which simply says, okay, that's cool.
You've figured out what the output shape needs to be.
I'm done.
I'm just gonna give you back that tensor without actually having done any of the computation at all.
So meta tensors are just tensors that don't have any storage associated with themselves.
They just, you know, like, they're just sort of like a abstract interpretation of the tensive just without the data in question.
So these are two new sort of features slash endeavors slash projects that have been going on in Pytorch.
Not every kernel is structured.
There's a bunch of kronos that you can port to structured if you want.
And I've got a very detailed RFC on the topic in the Pytruish RFC's repository.
And that's not really what I wanna talk about today.
I'm not gonna tell you really really about how structured kernels work.
So that I just wanna talk a little bit about the history behind structured kernels, And in particular, and the reason why I'm doing this episode, Angeliki Chordia asked me, Edward, you know, why did it take so long for us to do structured kernels.
They seem like a pretty simple idea.
This is not her words.
I but I'm I'm elaborating.
they seem like a simple idea, like, you know, of course, you don't want to write the shape checks multiple times in your CPU and CUDA kernels.
how come, you know, it wasn't done this way from the beginning? How come we didn't do it earlier? And this is actually a pretty good question because for me, I was you know, originally when I decided that I was gonna work on this, I thought to myself, oh, you know, I'll be able to wrap this up in a half.
I'll be able to port, you know, eighty percent of all operators.
life would be great, you know, what could possibly go wrong? Well, a lot of things.
So let's talk about that.
before I dive into when we started working on restricted kernels, it's useful to think about sort of what problems we're showing up for us in Pyturg development that sort of led to the idea that we actually need to invest some time on this.
And there are two, like, very distant causes that sort of causes some consternation and we didn't really act on that.
And then a more immediate cause and I wanna talk about the distant causes first.
So distant cause one was we were writing we were writing compiled passes for the jit, and they needed to do shape propagation.
And there's a problem.
Right? Which is that, like, hey, you know, you've got some input shapes and, you know, you're running an ad on them and you don't know what the output shape is.
How do you actually compute it? And so remember, like, you know, Pi torch as it is written mostly today and historically the way it's written, all the shaped checks, all the alpha computation, they're all sort of interleaved with the actual kernel computation that does the honest to goodness work.
So if someone came to you and they said, hey, know, I wanna know what the output shape of this they add on these two tensors of these sizes are, but, like, I don't want you actually doing in the compute.
I'd actually not have a good answer for you because there wouldn't actually be any way to call this code in the situation.
So what do people do? Well, you know, we could have done something like structured kernels, but we sort of wrapped around the problem by just being like, okay.
We're just gonna build we're just gonna write the formulas ourselves because, like, a lot of these operators, the shape calculations are really simple.
And, you know, what could possibly go wrong? So we wrote a bunch of shape you know, transfer functions that, like, you know, said abstractly what various operators did.
And these promptly fell out of date and no one uses them because, like, the coverage is really bad and a lot of them are wrong.
And they're wrong for really interesting reasons because it turns out that computing the output size of, like, an ad is actually really complicated in pie charts.
There's a lot of things that go into it.
because it's not just, oh, yeah, if the two sizes are the same, then I give you an output that's the same size because, hey, like, there is you know, broadcasting to worry about.
There is tight promotion to worry about if if you were cared about d types, which you often do in compile passes.
there's strides to care about if you're like doing memory loud.
Actually, the stride handling for like, you know, point wise operations is really really complicated because we need to answer questions.
Like, if I add an NCHW and an NHWC Tensor together, what is the up of layout? And, like, these are questions that are all resolved in the actual kernel stay.
And if you're just like someone, like, you know, like, who who you don't really care about these shape functions.
You're just trying to do some other work.
Right? That actually uses these shape functions.
You're not gonna spend the time thinking about all of the exhaustive error cases that go into this problem.
So okay.
So we needed some sort of shape past four jut, and we wrote a kind of crappy one and now no one uses it.
Actually, like, when people really need like accurate cheap information.
What typically happens is they just trace through a honest forgiveness real execution of the Python high George kernels running through the actual kernels in question, and then that gives you super accurate, you know, sizes and shapes and d dimers and layouts.
for everything that happened.
And then you can, like, just use that information directly.
Right? So, like, you just worked around the fact that you didn't actually have a function that you could've just call to find out what the shapes computed to be.
So this is like kinda like, you know, ugh, this kind of sucks, but sounds like we're factoring everything in PyTorch to, like, put the shake computation separately seems like a lot of work.
So, you know, I'm just a compiler developer.
I'm not gonna work on it, and so things stay like that for a while.
The second inkling we had that there would be a need for structured kernels was this, like, very old proposal called async CPU.
So what is async CPU? Well, you know, when we look at normal patterns programs, there's two devices that everyone uses.
CPU and CUDA.
Right? CPU is synchronous, you like say, okay, I wanna do an add and it goes ahead and does the add.
And then once the add's done, you get a new CPU tester with the result of having done the add.
Kuda is asynchronous.
I talked a little bit about this in my previous podcast about, you know, just enough to be dangerous in Kuda, Right? When you run a CUDA kernel, we actually run ahead and return to you immediately while the CUDA kernel is still processing.
And eventually, we we can keep queuing more kernels.
And only when we do a synchronize, we actually observe the result.
Well, there's nothing special about being asynchronous that requires it to only happen on CUDA.
And so if we are CPU, we can also just do a version of CPU that's asynchronous.
Right? So you like queue some work onto some thread pool and then the thread pool goes off and starts doing the CPU work.
and then, you know, you actually return immediately.
And so if your CPU computations are very beefy, then, you know, you might actually profitably reduce latency this way because you can keep you know, running your control thread along while, you know, you're chugging out the actual CPU computations.
So this was kinda cool and, you know, we were taught talking about this during the time, and there was a problem.
And the problem was, like, we really wanted to reuse all the existing CPU kernels.
We didn't wanna write an entirely new back end for an async CPU.
That would be silly.
Right? Because we got these perfectly good regular CPU terminals.
We just need to make them async.
But there was a problem.
If you want to return immediately after running, you know, queuing up the pool of work, you need to return a tensor.
And that tensor return needs to actually have all of the, like, you know, metadata, the sizes, the d types, the layouts, all that stuff.
Because we have a ton of code that assumes that I can, you know, without inducing a sync.
you know, access this information.
And in CUDA, this isn't really a problem because we like already did the copy paste from CPU kernel to CUDA kernel.
So, like, the CUDA kernels knew how to compute all the shapes while also asynchronously firing off the colonels because that's what the CUDA runtime dealt with.
But, like, if we were gonna do this entirely new ASIC CPU back end, it would be really silly if we, like, copy pasted every single CPU kernel and then, like, async if I did it.
Like, that would just be a terrible maintenance problem.
And so we couldn't implement async CPU because once again, there was no way to run commutations without without doing a a huge refactor of high torch.
And there weren't really that many compelling use cases for async CPU at the time.
So we just let that lie and, you know, it was just like, okay.
Well, we can't do this but maybe it doesn't really matter.
And so there were there was always other stuff to work on at the time.
The thing that actually convinced me that we needed to actually spend some time doing this manufacturing work was when I was working with Gram Wassey on his project called Lazy Tensor.
Lazy Tensers are this concept that, like, keeps coming back again and again.
And it's just, you know, instead of eagerly executing computations when you ask for them in your eagermode API, we wait.
we say, okay.
We're not gonna actually run these computations because maybe we will notice that there's a sequence of operations that happen and they can be fused together.
And then now I can actually, you know, use some fused kernel in this case and run a lot faster in this situation.
Lazy is different from tracing because with tracing, you just, like, run the entire computation through you you capture whatever the control flow was at that time and then you, like, compile the entire trace.
Laziness is sort of trying to be this more controlled controlled situation where you can run your code repeatedly and, like, you know, we'll keep lazily evaluating and then, like, doing the optimizations every time.
So actually, in theory, anything you could run-in eager mode, you could also run with a lazy tensor.
but you could actually pass it to some graph back end that does optimization on it.
It's it's very similar to tracing, but the difference is you do expect to run the eager code every time.
And, like, you know, if the trace is the same, then you reuse it otherwise, you know, you would compile.
XLA, by the way, in PyTorch is an example of a lazy cancer and PyTorch.
Okay.
So, Bram and I were working on this prototype.
Well, really, Bram was, like, doing all the work and I was, like, you know, advising as, like, someone who was working on Core Pytorch.
And besides, like, all the design problems that, like, lazy tancers entail and which would be a great story for another day on this podcast.
Something became clear, which is that, hey, when you do a lazy tester, you need to return a tester that sensor needs to have valid sizes and strides and d types.
But you didn't actually run your computation.
That was like, oh my god.
this is terrible.
This is exactly the same problem we run into, you know, third times of charm, let's actually do this.
And so I pitched structured colonels as this project and thus embarked on this year long journey to, like, actually bring structured colonels into PyTorch.
Why did it take so long to do structured kernels? Well, there's, you know, a really difficult problem whenever you wanna do any development in PyTouch, which is we have too many goddamn operators.
Like, we've got, like, So one of the things that I did before embarking on the stretch to colonel's project was to, like, try to taxonomize every operator in PyTorch.
And I actually, like, have a spreadsheet of all our operators, I, like, went through them one by one and tried to classify what kind of thing they did, what kind of shake computation they were.
And it was only like seventeen hundred operators this is slightly inflated because, like, when there was a in place and out of place and out variant, I counted these all separately.
But still, seventeen hundred offers.
That's a lot of operators.
You you actually have to do it.
And we keep adding new operators every, you know, release.
And so this number just keeps going up.
So, oh my god.
Like, how the heck are we gonna actually refactor all of this code? And it's even worse because remember, like, Pytorch came from Lua Torch which came from torch seven.
And so there's like this legacy c t h implementation.
And actually like we had already started a project for porting these krafty t h kernels written in c, written in this bastardized macro system and getting them into a more shiny, modern c levels plus.
And even to this day, we are still not done getting rid of all the t h colonels.
So, like, there's a lot of work and structured colonels like, refactoring kernels in this way would have been a lot of been a lot more work.
So, like, the first thing that I, like, had to grapple with was, like, how the heck am I actually going to, like, stage this change in a reasonable way so that we can, like, start partially migrating things while not having problems.
The second big problem that I ran into was tensor iterator.
So for those of you who don't know, Tensor iterator is the class in Pytorch which was responsible for implementing all of our urinary, binary.
And basically, all of our, like, you know, kernels that, like, you know, basically know how to operate on starlight sensors.
Tensorator is pretty cool.
It does a lot of interesting stuff.
It's also really, really, really, really complicated.
And, like, you know, if so remember when I was, like, how do you do add? Well, there's type of motion and there's, you know, layout, probably and there's all that stuff.
A lot of stuff is an intensity reader.
And it's like this big ball of code that like no one really knows how to refactor and I needed to somehow, like, not duplicate this code because, like, it's really complicated code.
I don't want two copies of it.
And at the same time, like, make it possible to use without, you know, running the computation, even though it's, like, embedded in this giant monolithic sensor editor class that, like, I have no idea how to do.
That, like, took I don't know.
I think it, like, took two months to figure out a reasonable design for stretchy curls that could actually deal with this involving, like, basically, I added a virtual method to Tensor iterator got invoked once it had actually figured out what the sizes and the shapes and the d types were.
And then overrude it to call into the structured kernel machinery.
The the technical details are important.
But, like, basically, big blob of legacy code And originally, I was like, I'm just not gonna solve this problem because, you know, temperature is too complicated.
Someone should just rewrite it.
But, like, add and solve and all these really important operators are sensor editors.
So I needed to, in fact, figure out some way to actually solve this problem.
So yeah, so that all took a while and we're still not done.
There's still a lot of kernels that need to be ported to structured.
But we're in a much better spot right now.
There's a lot of work going on porting, criminal construction, and Pytorch.
We're getting better and better coverage.
We're hoping to hit covering all the operators that XLA supports.
That's a really decent chunk of operators.
And I don't know.
I'm pretty optimistic about the project even though, you know, It's like sort of sucked up my time and energy for for a year at this point.
That's all I really wanted to say about structured kernels and meditensors.
meditensors, by the way, really simple.
Right? But how are you gonna test them? And like getting testing to work on them was also a project? But but I'm out of time.
I'm gonna leave you all here.
Thanks all for listening.
See you all next time.
.
