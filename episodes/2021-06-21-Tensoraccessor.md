---
layout: post
title: "Tensoraccessor"
date: 2021-06-21
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Tensoraccessor

Hello, everyone, and welcome to the Pytog disturb podcast.
Today, I want to talk about Tensor Accessor, a way of accessing elements for Tensors when the dimensionality and d type are known.
In previous podcast, I've talked a little bit about the API design principles behind our c plus plus API.
And one of the Calvert's characteristics of Tensor in c plus plus is that it is completely type erased.
You get to know you have a tensor, but you don't know what its d type is, you don't know what its dimensionality is.
Doing things this way makes polymorphism easy because you don't have to write templated code.
but this type of razor has costs, namely performance costs.
And it's for this reason that like other c plus plus libraries that do tensor computations often do, in fact, encode this information directly in.
So for example, iGain, a very well known and popular library known for its fast implementations of kernels, uses fixed dimensions inside the tensor itself.
So what's the problem? So the problem is when you don't know what the dimensionality of a tensor is and what you don't know what the d type of a tensor is, In order to do operations on this sensor safely, you have to do dynamic checks.
If you want to you know, retrieve an actual element, like an honest, goodness, single element, from the tendering question, you are going to have to say you're gonna wanna fetch it into some d type like float or double.
And technically speaking, unless you provide an unsafe API, you need to test that the d type of the tensor actually matches what you want to read the element out of.
Otherwise, you can read out complete garbage silently in this situation.
And so if you think about, like, the data pointer API in Tensor, that actually does in fact do a d type check whenever you do this.
Similarly, when you want to index into a tensor, well, if you don't know what its dimensionality is, then you have to actually write code that knows how to loop over all the indices you wanna do and multiply with the strides in question.
And so, you know, because you don't know how many dimensions there may be.
Right? So you can't write a fixed index calculation in this situation.
You had to write a loop that can handle all the sizes in question.
And so if you're a tensor iterator and, you know, you're doing a lot of hard work to make sure you can write an algorithm and then work with arbitrary dimensionality, That's cool, and Tensurator is kind of complicated, but it does that all for you.
If you're just writing a good old fashioned kernel, you probably don't actually need this generality.
You probably only are writing kernel that only works for some set of dimensions, etcetera.
So if you want to do lots of low level manipulations to data in your tensor and you don't want to go through all the overhead that tensor would be.
And, yes, you could write a loop over a tensor and then say directly x open square bracket index, close square brackets equals blah, but trust me, you really don't want to write your kernel that way.
It's really really slow because each of these indexing operations is actually gonna give you another tensor back.
Even if it's actually a scaler, it's a single number.
You're gonna do an entire dynamic allocation and that's the case.
So if you want to do this sort of thing fast, what do you do? And so the sort of like very easy way to handle stuff in the situation is to get out a raw pointer and do the manipulations on it.
It's the obvious thing to do.
Right? Because, you know, what are CRAs? Well, CRAs are okay.
They're not exactly the same thing because the, like, type size is different.
But, like, a c array is basically a pointer to some memory and then, you know, you just operate on the memory.
So what do you do? So if you have a tensor object, you can call data pointer to get out of raw pointer.
That is going to give you a fixed d type.
So it's going to check what the d type is.
And then you can just poke it, you know, index into it, the same old way, you'd have index into any sort of array, and, you know, work with the data in the tensor that way.
There are a few implicit assumptions that are going on when you do things this way.
So one is that you are probably assuming that the data in question is contiguous.
Why are you probably assuming that the data in question is contiguous? It's because handling strides is actually a pain in the ass, and so you probably aren't going to go through all the rigmarole of doing strides exactly correctly with the pointer in question if you do it this way.
You're probably more likely to just, you know, directly compute some linear index or, you know, you you have a one dimensional tester and you just can index directly.
you're not gonna handle that.
So whenever I see kernels that are written directly using raw data pointers, I usually assume that they are assuming contiguous inputs.
The only exception is if I'm, like, identifying out to some external library where they have to take a data partner and then they take them into strides as the other thing in question.
So well punters very easy, but typically only used for contiguous sensors.
But what if you wanna do some accesses? and you happen to know that you want to handle strata things directly.
You don't wanna actually go through the process of taking a possibly non contiguous sensor, you know, allocating memory to contiguous fire it and then run your kernel on it.
conteguifying a tensor by the way, you know, is kinda slow and it uses up memory.
So if you can just directly fuse your computations directly on the input sensor, that can save you quite a bit of computation.
And this is what tensor accessory is for.
So what is tensor accessory? Tensor accessory is a specialization of tensor where the d type and the dimension of your tensor are fixed.
However, we don't make any claims about the sizes or stripes.
So the sizes and stripes continue to be you know, sort of built into the class in question.
And so if you look at what the representation of Tensor Accessor is It's very simple.
It consists of a data pointer.
It consists of a deep the sizes and it consists of a pointer to the strides.
In fact, tensor excessers are really lightweight and they don't involve any dynamic allocations.
because they're also non owning unlike regular tenses, which, you know, guarantee that the data pointed to stays live.
the lifetime of the tendering question.
They're non owning, so they're really cheap to allocate.
And lastly, right, as I said, they have statically known d type and demo dimension.
The statically known dimension is important because it means that we can implement index calculation without doing any loops.
Right? So like how it's implemented in PyTorch is it's actually a recursive template where, you know, like, the tensor accessor for n is computed by doing the tensor excessive for n minus one, and then, you know, adding on the indexing for the last dimension that we're processing.
And then there's a base case for tensor tensor accessory one d tensor where you can just linearly index in that situation.
By the way, this is a nice thing about being in c plus plus.
In the battle days of t h, these fast indexing operations were manually specified for every dimensionality So there's like a one d fast index, a two d fast index, three d fast index, four d, and so forth.
Tensor accessory also optimally supports declaring the pointer as restrict.
What that means is a pointer that's restricted is guaranteed not to alias with any other pointers that are in scope.
and sometimes that can unlock easier compiler optimizations.
We use this very rarely, but it's often useful in CUDA where non aliasing is a useful guarantee.
There's also a variation of tensor accessory called pack tensor accessory.
So I said tensor accessory is a non owning So it, you know, contains a pointer to the sizes which are actually stored in the good old fashioned traditional tense sequence in question.
And a pointer to the strides which are also stored in the old tansering question.
But sometimes we wanna send these, like, you know, raw pointers plus metadata to CUDA current and with CUDA kernels, you have to send all this information.
If you have this pointer to some random CPU memory, well, of course, your CUDA kernel is not going to be able to access it.
because CUDA Crohn's can only ask us CUDA memory.
So you have to pack everything up into the parameter list that, and, you know, is gonna be sent along with the CUDA kernel launch.
and packed tensor accessory basically just packs all of the sizes and strides along with the data pointer directly into a, you know, compact representation.
Remember, it's fixed dimensions.
So we can allocate precisely the amount of fields we need to actually do this sort of thing.
And then, you know, you can ship them all to CUDA all at once so that CUDA can then use these to compute the indexing.
And for CUDA, like computing indexing is pretty cheap because, well, you know, it's CUDA and you've got tons and tons of little processors that are doing these computations in parallel.
you're more likely to get hosed by, you know, memory bandwidth because, you know, you're accessing stuff all over the place.
So let's just step back a moment.
So suppose you're writing a kernel in PyTorch and you need to actually do some manipulation on the data in question.
well, there are a few things you can do.
Right? One is you can, like, directly use the Tensor API, and that's okay if you're gonna just call a bunch of other, like, sort of accelerated operations.
This is a bad idea if you actually wanna do like element by element operations.
Then there's raw pointers which are sort of the easy and obvious way to do things.
but they don't do any of the bookkeeping for strides for you.
So usually, people only do them when they assume contiguous inputs.
So you'll see, you know, run contiguous on the input and then get out of raw pointer and do something with it.
And finally, Tensor Accessor knows about sizes, knows about strides, and so it can let you do fixed dimensionality indexing on tenses that might have, you know, wacky layout without having to do the, you know, sort of indexing math all by yourself.
It's handled for you automatically under the hood.
One current limitation of tensor accessory is that we don't define any operators on them.
So once you go from a tensor, to a tensor accessory.
You can't like a view the tensor and you can't, for example, reshape it.
Actually, we have an old version of packed tensor accessory called THC device tester.
That was part of the THC library.
And this tester did have a bunch of operations on it.
And there's no reason you can't implement these off operations.
In particular, anything that's a view, really good math for tensor accessory.
Right? Because tensor accessories are not owning anyway, so you're usually just fiddling around with the size and starrettes.
This would be a really nice feature to add to PyTorch.
No one has really done it yet, but it would be useful.
Another thing that I've been thinking about is sometimes we get to know that a tensor is some dimensionality fairly early in the stage of a sort of a multi operator composite function.
And it would be nice to not have to keep, you know, doing the dimensionally check locally at the kernel site whenever you need to use it.
Like, it would be nice to, like, do it once and for all at the beginning of a composite kernel and then pass on this information statically to the kernel you're gonna call later on.
Of course, there this is rife with difficulties.
Right? Like, if you want to be polymeric over the d type in this way, your kernels have to be templated.
But it's a kind of interesting problem about, you know, like, how can you write code that doesn't need to be templates instantiated, but can still propagate type information like this.
And so maybe, you know, having some sort of, like, fixed dimensionality, but the d type isn't fixed, tensor type might let you do that.
But I don't know.
That's something that I've been thinking about.
That's everything I wanted to talk about today about Tensor Accessor.
Talk to you next time.
.
