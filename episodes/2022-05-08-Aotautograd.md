---
layout: post
title: "Aotautograd"
date: 2022-05-08
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Aotautograd

Hello, everyone, and welcome to the Pritchard dev podcast.
Today, I have Horace Hoe with me who is going to come and talk about AOT Autograph.
a system that is in functorch, which lets you capture both the forward and backward traces of Pytorch operations.
And then you can send them to a compiler and then get back a compiled kernel and then stick them back in your Pytruch program just like any other old function that's available.
So, Harris, can you tell me a little more about what AOT Autograph is? Yeah.
So AOT Autograph is kind of just as I said is essentially a compiler integration point for PyTorch, yet another one.
So kind of the main premise behind AOT Autograph is that we want something that makes integrating compilers into PyTorch training easy.
So we have, you know, other APIs like, you know, torres dot FX or Torrescript.
But one of the things that makes integrating against training difficult is that they kind of have these, you know, very specific APIs in the case of tool script or they just, you know, don't support Autograph.
in the case of FX.
So the premise behind AOG Autograph is that we want to provide an integration point that makes integrating compilers in their training seamless and basically as easy as integrating compilers for the purposes of inference.
And the fundamental reason why this should be doable.
Right? Is that the operations during training are node are just tensor operations.
And these are exactly the same terms of operations that occurred during inference.
The shape may be a little bit different, but the actual operation should be the same.
So as long as we can represent the operations that occur during the backwards pass in just a straight line, like, graph of operations, we should be able to use those with arbitrary comparators.
So to actually achieve this, AOT Autograph actually has to do a bunch of things, which I guess normally we would think of as separable components, but AOT AutoGuard just puts them all together in one package.
So for example, you mentioned the other tracing mechanisms like torch script or FX.
So AOT Autograph does come with a tracer.
Is that right? Correct.
So one of the aspects of AOT Autograph is it uses this tracer built on top of a new mechanism called torched dispatch.
And one of the towards dispatch is kind of this new accessibility point that probably, you know, it could do with its own postcodes episode.
But kind of what towards dispatch does is it's a multiple dispatch integration point that sits below the dispatcher.
So unlike, you know, something like torch dot FX that sits at the Python level or something like Jira trace, which sits above Autograph, towards dispatch sits below all of that and therefore allows you to capture autograd.
So that's kind of, you know, the tracing mechanism that AOT autograd leverages capture the forge paths and backwards paths? That's right.
So we are basically able to run both the forwards and backwards and trace all of that, including the code that normally is running in c plus plus.
But once you're done tracing all of that, there's still more stuff AOT Autograph does.
Is that right? That's correct.
So one of the tricky things about so one of the things that AOT Autograph or like one of the premises here is that tracing is fundamentally a pretty good way of capturing machine learning models.
And the reason for this is that tracing most users code ends up being fairly lacking in, like, dynamic control flow and things like that that make tracing gold.
And a lot of, like, what you need to do with tracing is like, well, a lot of what tracing is able to do is it's able to eliminate that, you know, like, Python data structures or, you know, like, weird ways that users write code, like, they might use Lambda's, they might use other data structures, and it captures that.
So, you know, that's kind of what tracing is good for.
But the problem is that oftentimes, there's a lot of things that you know, Brake tracing.
So for example, users might want to log their tensors.
They might want to, you know, branch on you know, like, user input, they might want to, you know, branch on, like, the loss and, you know, do different things.
And so what we want to do with the AOT autograph is we want to be allow you to apply a compiler to an arbitrarily small subsection of your program.
And this is not Like, this is not naturally fitting into the tracing paradigm because, like, if you capture a sub part of your model, The forward spreads and backwards paths do not actually run at the same time.
So it's not like there's like a single function that you can trace.
And so what we need to do is we need to capture the forwards and backwards fast simultaneously by pretending it's a single function And then we need to do something else to be able to split the force and the backwards pass into two separate graphs that we then run at different times.
So just to emphasize here, normally, you think of tracing the entire model, but with a o t autograd, you're just tracing a little piece of it or or the entire thing if you can manage it, but more frequently, it's just gonna be a little fragment of it.
And that bit needs to be its own microcosm getting it forward and backwards.
And then, I guess, a o t autograph, the name autograph in it comes because, in fact, the main thing it does is it creates a custom autograd function that wraps up the forward and backward that can interoperate with the rest of your eager code.
That's correct.
Yeah.
You mentioned this this cut thing.
What's that? So, like, once you've traced your joined forwards and backwards graphs, you now need to convert this, like, single joint graph into two graphs, like, you know, one that runs in a four inch bass and one that runs in a backwards bass.
And It might be clear that, actually, this is there there's actually some leeway in how you're willing to do this.
So there's, you know, there's some strict dependencies such as operators that need to be in the force password that need to be in the backwards pass.
But there's other operations where you have a choice of whether you wanna put it in your forward pass or backwards pass.
And so this choice actually ends up mattering in certain cases.
So you might imagine that if you, like, if you put an operation in the foreach as a foreach artist, this might expose more fusion opportunities.
or other things like that.
And so one of the things we've kind of figured out is that one of the most important optimizations you can do here is something called Rematerialization, also often known as gradient checkpointing.
So what we so we've kind of come up with an approach that minimizes the memory transfer between your forward class and backwards class.
and this is kind of done using a mid cut algorithm that's, you know, allows yeah, using a mid cut algorithm.
And the kind of one of the new things about this approach is that in combination with a fusing compiler, this allows us to improve both the runtime as well as the memory usage of the function.
So if I understand you correctly, what happens is we trace out the forwards, we trace out the backwards.
Actually, When I run traditional PyTorch Auto Grad, there there is a choice made, right, about what I compute in the Ford's and what I compute in the backwards.
But that choice is fixed.
It's whatever my derivative formulas were implemented.
But then with AOT Autograph, once I've got this trace, I've got both the backwards and the forwards.
I can I can basically renegotiate the boundary in that case.
Is there a really good example of someplace where this is really profitable? Yeah.
So I I think a pretty natural example is let's imagine you have a sequence of operations like a cosign.
you're just calling, you know, cosign on a tensor, you know, five or ten times.
So if you think about the automatic formula for cosign, it requires saving the saving, like, the output sensor of your cosine operation.
And so if you call cosine ten times and you're going to save the output sensor, you're gonna save ten different outputs.
Right? because, like, you know, the way Autograph works is you apply the Autograph Formula to each operation individually and then you, you know, like multiply them together and apply the chain rule.
So if you just run Pritchard's eager autograph, you'll end up saving ten different tensors between the four spats and the backwards spats.
But instead, often what you should instead do is you should just not save any operations in your force pass And so, therefore, like, you just get, like, a straight line graph in your forecast pass that doesn't save anything.
And then in your background pass, you should simply recomute your forecast pass in your background pass.
And so this allows you to reduce instead of saving ten tenses for your force pass, you only save one.
And instead of really ten tenses for your backwards pass, you only read two tensors.
And because these cosine operations are what's known as bandwidth bound operations, fusers can fuse them and make them Like, you're you're kind of dominated by the memory you're running and not the actual computation.
So because you're reducing the memory usage or like the memory reads and writes, we can actually improve both the runtime as well as the memory music.
And AOT Autograph does this today? AOT Autograph does this today.
Yeah.
That's actually pretty surprising to me because I remember when I first read about the min code algorithm, I just imagined this was, you know, sort of changing the sort of what tensors we say like, there would be some intermediates, and we would choose to save some, but not the other and then you just move the boundary around.
But fundamentally, the computation wouldn't be changed.
So I guess, that's not exactly what you're doing here.
Right? There there's also rematerialization going on.
How how does the algorithm figure out if something should be rematerialized.
Right.
So I I think the the way I think of this the way I think of this is that basically the value we really care about is the gradient of the input.
Right? So, like, your the only reason you're doing your force pass is to compute the gradient.
So perhaps, like, the right way to think about this is that give it let's say, like, you know, you're given both the inputs to your force pass and the inputs to your backwards pass.
And you're allowed to save any arbitrary values such that computing like your the gradient input is the easiest.
So for example, you can you can, you know, compute it from your let me just cut in for a moment.
You said gradient inputs twice, but actually you're given the inputs and the grad outputs.
and we wanna compute the grad inputs.
Alright.
Yes, sir.
That's correct.
And so, like, one strategy you might do, right, is you might take both like, you might compute your the gradient of your inputs by taking the inputs to your forecast as well as your grad outputs.
And so this corresponds to basically, like, recomputing the entirety of your force pass during your backwards pass.
But you can imagine that other strategies, for example, perhaps you're doing a Marvel in your forecast, you might want to start compute you might want to start computing later down in your graph and, you know, skip the extra metall during your backwards pass.
So, like, this is kind of you're doing like a min cut, not exactly to partition the two graphs, but kind of to decide what computation are gonna perform in your backwards path.
In other words, the forward's computation actually always is the same no matter what So so we're not really partitioning the graph in that sense.
Or no.
So the Forest Pass is kind of the way I think of it is is that it's implicitly defined by what you choose to save for your backwards pass.
And because, like, the thing we're trying to minimize here is memory bandwidth costs.
And any each input you need to save in your four's pass corresponds to one input you need to read in your backwards pass.
So luckily, minimizing minimizing memory bandwidth actually ends up being symmetric for both in force pass and in backwards pass.
So that's kinda like one non obvious thing that that makes us easier.
So I misunderstood.
So so we are gonna change what the Ford passes, but we are going to sort of move stuff over into the backwards pass if we think it will be profitable? That's correct.
Yeah.
Okay.
So we've talked about how AOT Autograph, you know, traces through our code, gets the backward traces, and then we'd also talked about how we split them up and then eventually put them in a autograph function so that they work with eager mode.
But of course, we need to actually run a compiler on these traces to, you know, do something useful with them.
So can you tell me a little bit more about, you know, how how a or two algorithm works with compilers? Right? So so what ASE autograph does.
Right? So we trace things out.
And we actually trace things out into this kind of, you know, like, standard high torch graph format, called like a you called torch dot FX.
And then we simply take this FX graph and we pass it to a arbitrary compiler.
So for example, one thing we might do is we might take the suffix graph, we might superscript and then we might pass it to, like, a transcript of fuser such as MBFuser or NMC.
So one of the, like, complexities here So, you know, like, the kind of picture, right, is that if you have a compiler that works for inference, You this like, AMG Autograph allows you to also apply that compiler in your background space.
And one of the Alright.
That's right.
Because we don't actually pass it.
We don't expect the compiler to do differentiation.
We just pass it the forward and the backward separately.
Right.
But, like, one of the one so that that's kinda kind of the pitch.
But one of the areas where that's not quite true, right, is that there are certain operations that only occur in your backwards pass, that never occur in your forwards pass.
So for example, Pyturbine has operations like ten h underscore backward, that, you know, are used to compute, like, you know, backwards from over ten h.
But oftentimes, compilers won't have support of this operator because they never shows up in their Forage Pass.
So one of the things we kind of do as part of AOT Autograph is we've written a bunch of these decompositions to basically rewrite these operator, like, you know, ten h backward in terms of other more common operators that compilers can't use.
So now to sum it up, AOT autograph is a tracing mechanism.
It's a min cut.
algorithm, and it's also a number of decompositions.
That's a lot of things fact into one box.
Right.
So, I mean, like, one way to, you know, view AOT autograph is, like, it's this specific kind of product where, you know, providing to, you know, and trying to improve performance.
But I I think another way of viewing a t autograph is it's kind of meant to be a, like, extensibility point for Pytorch.
So there's, like, many things that are there's many things in Pytorch that like, are hard to do because we're in eager mode.
And AOT Autograph is basically, like, an easy way of, you know, getting a graph for your force pass and your backwards pass that allows you to do arbitrary things, you know, like rewrite them or, you know, reinterpret them in other ways.
Yeah.
So for example, you know, like, it's really easy to, like, change this very much like, this mid cut rematerialization approach with a different algorithm.
So for example, you might want to, like, save more memory at the cost of doing more compute or things like that.
And that's kinda like one of the things that we're trying to support.
So that's a lot of cool stuff.
And if I understand correctly, there's a lot of things that we also want to do with IoT Autograph in the future.
Can you tell me about some of them? Right.
So one of the, like, the most, I guess, obvious things, right, is, like, we've kind of implemented a couple of authorizations such as rematerialization as well as, you know, hooking up with the operator fusers.
But, you know, there's way more optimizations, like, still left that we haven't really even, like, touch the surface on.
So for example, one of the things that you might wanna do is kind of layout planning.
Like, you you might want to, you know, change the layout of your operators so that, like, you know, the mammal or the cloud has, like, a more favorable performance.
And kind of or, you know, other things I might do, you might wanna do, like, memory planning.
And one of the kind of interesting things here about AAT Autograph is that The setting that we're operating in is kind of often fundamentally different from what many like, what a lot of the people in, you know, literature or have kind of historically looked at.
In that, a lot of people usually kind of assume that they just get, like, an entire graph in a single or like they get the entire model, you know, forwards and backwards in a single graph.
But kind of the what we believe here is that this will early, you know, Like, what we kind of believe in PyTorch.
Right? Is that, like, this is not really true.
And then a lot of times, users want, you know, the flexibility of PyTorch, and they want control flow and things like that.
So a lot of, you know, things like layout planning or, like, you know, memory planning become trickier when it comes to, like, operating in this setting.
So that's kind of one of the things we're thinking Another one of the things we're thinking about is that in some sense like AOT Autograph is pretty inspired by you know, Jackson's like Jit and, you know, how is composable with Jax.
So, you know, you can apply Jax dot Jit in an arbitrary location and, you know, a composable autograd or a composite of v map and things like that.
And, you know, currently, we also have things like v map.
And AOT Autograd, unfortunately, currently does not compose with those.
So although you can use AOT Autograph to compile, you know, things like VMAP.
And, you know, we've used that for, you know, things like Jacobians or hessians.
It does not allow you to compose in the other direction.
So you can do a anti autograph of the map but you can't do v map over AOT autograph.
So, you know, kinda figure out composability in that manner is kind of one of the other things we're currently thinking about.
The way I think about this problem of, you know, running a v map over an AOT autograph is In some sense, it's not AOT autograd anymore, but it's AOT.
Everything that PyTorch supports So, you know, that includes Autograph, which we do support right now, but it also includes batching and functionalization and all of the other on user platforms going on.
Harsh, do you agree with this point of view? Yeah.
So annot annotate is kind of like the current name.
But in the future, yeah, perhaps they'll need to be named something more generic.
Alright.
Well, that's it for our time today.
Thank you very much for joining.
Hi.
Thanks for having me on.
Cheers.
.
