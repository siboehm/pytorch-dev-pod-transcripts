---
layout: post
title: "Half Precision"
date: 2021-09-09
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Half Precision

Hello, everyone, and welcome to the PyTorch dev podcast.
Today, I want to talk about reduced precision floating point formats, namely float16, aka half-precision, and bfloat16 brain floating point.
Float16 and bfloat16 are important alternative precision for floating point numbers as opposed to the ordinary 32-bit floating point representation, which are often used in deep learning applications to speed up computation in cases where the extra precision afforded by 32-bit floating point numbers is not necessary.
I'm not really going to have time to give you a complete low down about everything there is to know about IEEE floating-point numbers or how these formats are set up.
But I do wanna give a little bit of working knowledge about some of the important points of floating point formats and also, you know, how they affect how we write code inside PyTorch.
Because something that you very often have to do, for example, when you're writing a kernel, is you'll write a normal implementation, the normal mathematical way for 32-bit floating point and for 64 floating point.
But then for half precision, you need to do something special.
And we'll talk about why you often need to do something special in these situations and what kinds of things you have to pay attention to.

So to start off, let's talk a little bit about floating point numbers what they are to understand why half precision does something a little unusual with floating points in the normal sense.
So floating point numbers are a way of representing decimal numbers because if you're, you know, familiar with normal computation on computers, you know, we love integers.
We use integers to represent everything.
But sometimes, some things can't really be represented as integers.
Right? Like numbers with decimals on them, for example.
And so the float in floating point numbers refers to the fact that we change the precision by which we're representing numbers depending on how large the number is.
To understand what I mean by this, let's think about a situation where you don't care about floating point precision, namely storing currency.
So, you know, in US dollars, you have a number of cents and you have a number of dollars.
So, you know, I may have $10,46.
And there's always some amount of sense associated with any number.
No matter how big the quantity of dollars I'm talking about, like a million dollars or a billion dollars, there's still only is ever two decimals of precision that I need to record the number of cents in question.
There's never like subcent quantities in typical monetary transactions.
So this is a prototypical example of a fixed point number where you want to fix the decimal point at 2 digits of precision no matter how big the number in question is.
Of course, if you're doing something like doing a measurement between how far you are between two cities or, for example, representing a neural activation and a neural network.
If your quantity is in the millions, you don't really care about those two digits of precision after the decimal point.
So the idea behind floating point is that you don't have to, store significant digits based on where the decimal is, you let it float and you just store a fixed number of significant digits and just what those digits are depends on how big your number is.
Right? So if you're talking about a million, then you might store significant digits for the millions, the hundred thousands, the ten thousands.
But if you're storing something like 1, then you might store the, you know, first decimal place and a second decimal place and third decimal place.
It floats along with you when you have the number in question.[^floats]

So given this basic specification of floating point numbers, there are basically two major parameters that you can vary when you're defining a floating point representation.
Right? You can say how many bits you're going to use on the significand, aka the significant digits that are in your number and how many digits you're gonna devote to representing the exponent, which basically just says how big the number is.
Right? Are you talking about ones or thousands or millions or billions.

And so we can use this to sort of understand what's going on with 32-bit floating point numbers, 16-bit floating point numbers and also brain floating point because they all actually have the same semantics, just different settings for these parameters.[^floatingformats]
So the significand for 32 floating point numbers is 24.
So that's a lot of digits of precision.
And so one of the observations that drives lower precision floating point is that, well, you don't actually need all those significands.
So 16-bit floating point numbers only have 11 bits of significand and bfloat16 only has 7 bits of significand.
If we talk about the exponent instead, while 32 floating point numbers have 8 bits of exponent, 16-bit floating point numbers reduce the amount of exponent you have.
So you only have 5 bits of exponent and bfloat16 actually keeps the number of bits for the exponent the same as 32 bit floating point numbers.
So another way to, like, think about the difference between float16 and bfloat16 is float16 sort of was like, okay, well, we need to chop off 16 bits from our representation to, you know, reduce it in size by half.
We're gonna chunk some of it off of this thing if we can.
We're gonna chunk some of it off of the exponent, and then, you know, we have a nice balance.
Bfloat16 was like we want all of the exponent bits.
We want the same, what we call dynamic range.
The same, you know, max and min values you can represent in floating point numbers, and we're willing to chop off tons and tons of actual precision off of the actual, like, you know, digits in question, the significand to get it.

So why use half precision or bfloat 16 numbers? Well, as I mentioned before, they use half the space of memory that a 32 bit floating point number uses.
So this has a number of implications.
Right? we need to store the values of tensors in memory.
And so if you can store a number in half the space, well, you've basically doubled the number of parameters you can store in your model.
And furthermore, not just you can store more numbers in your RAM, But when you're actually like loading up this data into your processors to actually compute on it, well, that's half as much memory bandwidth you need in this situation as well.
And oftentimes, one of the primary costs of doing deep learning inference or training is just getting the freaking values out of memory in the first place.
And of course, if you only need to compute on 16 bits of data, instead of 32 bits of data, that means less silicon and you can, for example, vectorize more easily for the same amount of silicon.
Now it sometimes you know, the memory benefits I would say for half precision are the primary benefits.
And the computation benefits do help sometimes, but also sometimes they happen not to matter.
And we'll see an example of this when we talk about CUDA support for half precision.

So let's talk specifically about half precision for a moment.
So what are some things to know about when you are writing code that needs to operate in half precision.
So one of the, like, things you first figure out that's very, very obvious, is you are way, way, way more likely to overflow your floating point number than if you're doing 32-bit floating point numbers.
A float 32 can store values up to 10^38.
That's 38 zeros.
I don't even know what quantities can go that high that I normally deal with in a day-to-day basis.
In contrast, a float16 can only go up to 65504.
That's it.
If you go much higher than that, it'll just go to infinity in float16.
So yeah, gotta be super super careful.

Because the dynamic range of hack precision floating point numbers is smaller, when you want to do training with networks and you want to use a half precision instead of float32, you often need to tune your hyperparameters differently because you need to make sure you don't actually go outside of the dynamic range supported by half precision.
One of the most common ways people use half precision is in fact not by making their entire network operate only in 16-bit floating point numbers.
That's often just too much.
It's like too little precision and your dynamic range is just gonna get messed up in all situations.
But instead by using something called automatic mixed precision, which just says, well, there are some operations that are very unlikely to go outside of the dynamic range you want, and we'll only cast to float16 and make use of the benefit, the lower memory usage, in those situations.
It also helps that automatic mixed precision is super easy to use.
You literally write your network as if you're writing it for 32-bit floating point numbers and then you just turn on a flip switch that, like, automatically switches it without you having to do anything.

Half precision has been around for a while and it's been available in NVIDIA CUDA. 
There's actually really no silicon for doing half precision computations on Intel CPUs.
And so you're most likely to see use of half precision inside CUDA programs.
But actually, there's a little bit of nuance to this which is that you might imagine that, like, you know, you put your tensors in half precision and then you do operations on them and you'd expect to see, you know, actual, like, half precision silicon being used.
But in fact, in PyTorch, we don't use any of CUDA's half precision intrinsics which would let you actually use the half precision operations directly in the hardware.
Instead, we convert everything into floating point numbers and do the computations at higher precision.
Why do we do this? Well, it's because for many of our operations that we've implemented, for half precision, they are in fact not compute bound, they're bandwidth bound.
We spend more time reading the data out from memory than we do actually doing the computation on it.
And in these situations, it doesn't matter if we waste time doing conversions to and back from floating point because you know, we're still waiting on the next block of memory.
And so we can just, you know, do things in higher precision.
And so a lot of computations in CUDA operate at this higher precision internally, only converting back to float16 when you need to write it out back out into memory.
Remember, this is still a win because you're using half as much memory, using half as much memory bandwidth.
So what you would typically expect is for a computation on half precision to be twice as fast as a computation in 32-bit precision and that's just because you're literally reading out half as much memory.

That being said, in some situations, you are somewhat compute bound.
A good example of this is when you're doing matrix multiplies.
And so when you do matrix multiplies there is this thing called TF 32 that newer NVIDIA GPUs implement where they do the multiplies and matrix multiplies at an internal format.
And in fact, they don't do it in half precision.
They do it in a special precision that is 11 significant digits and 8 exponents.
So like a combo of float16's precision and a bfloat16's dynamic range and this happens entirely internally.
So you don't see it.
You're just feeding in float 32s and getting out float 32s but it makes things run faster and, you know, you hope that the numerics don't change too much.

So to summarize, in half precision the dynamic range is way, way smaller.
So you're mostly likely to see people converting a half precision at very, you know, localized spots in their code where they know they don't actually need that level dynamic range, and you mostly only ever see half precision in CUDA on NVIDIA GPUs.

Okay.
Let's talk a little bit about bfloat16.
So as I said bfloat 16 is they just took float32, chopped off enough significant digits until they, you know, could fit in 16 bits and they kept exactly the same dynamic range.
So bfloat16 is actually very easy to emulate.
Right? Because you can use normal 32-bit floating point hardware to run it, you just, you know, sort of zero out all of the digits that are below the level of precision that bfloat16 would have given you and then just run the normal float32.
So people did a number of experiments with it and showed, hey, you know, bfloat16 is great because, you know, we got rid of all of those, you know, pesky, like, very fine detailed digits in the numbers.
And it turns out it didn't matter at all.
Like, you know, our model still converged because we weren't actually making use of that precision in any good way.
And so bfloat16 has shown up in a lot of places.
True to its name, brain floating point, It was originally designed by folks at Google for use inside the TPU.
But since then, it's shown up in a lot of places in particular, on the latest Intel CPUs, starting with Xeon, there's actually silicon for doing bfloat16.
So unlike half precision, which only ever usually shows up in CUDA, bfloat16 shows up in a lot of places.
It shows up in GPUs.
It also shows up on your CPU.
So if you're probably looking for some lower precision training, it's probably gonna be bfloat16.
In fact, Intel has been working with us to extend, automatic mix precision to support bfloat16.
So originally, AMP was something developed by NVIDIA for CUDA only for half precision and Intel is, you know, giving us a patch that turns it on for CPU and does exactly the same thing, except using bfloat16 instead of half precision.

Unlike in the CUDA situation where we were typically memory bound, we are often compute bound on CPU.
And so sort of the silicon we're using is in the AVX512, you know, vector instruction set, see also my previous podcast about vectorization.
And there's just, you know, a lot of built-in support for actually doing these computations fast.

Okay.
So I've told you a lot of facts about float16 and bfloat16.
What does this matter if you're doing code in PyTorch? Well, it mostly only matters if you're writing kernels.
And so when we write kernels in PyTorch, we typically try to write it in a generic way that works for any type in question.
Right? So typically, it's templated so that you can do it in float and you can do it in double.
And for most use cases, float versus double doesn't really matter.
You can write the same algorithm in all of these cases.
But when you have float16 or bfloat16, now you actually have to pay attention to how you're doing your internal computations.
And in fact, we have two concepts for like basically internal computation types, which are important when, like, you know, using the low precision floating point would result in catastrophic loss of precision, and you'd basically get wrong results.
So the first concept is the AccT type template, a Acc underscore type.
What this does is it gives you an accumulator type corresponding to the number in question.
So for example, if I had int8, the AccT type of int8 is int64.
Because if I'm, you know, summing together a bunch of 8 bit integers, I will very quickly overflow 8 bits.
And so we do the accumulation in 64 bits so that we can actually, you know, get the real value in the situation.
Similarly when we do accumulations on half precision floating point numbers, we need to accumulate them in 32 floating point because as I said, you're really likely to overflow 65000 if you don't actually do this at a higher precision.
This is very very common.
Right? Like I mentioned matrix multiply, using this TF 32 thing, they only do that for multiplies.
The accumulate still happens in 32-bit floating points.
So like the idea of doing accumulate at a higher precision is common all over the place.
We don't accumulate in double precision for float on CUDA because double precision hardware is really, really slow.
But in fact, on CPU, we still we act type goes to double in this situation.

The other concept we have is op math, and that just says what the internal computation type we're gonna do.
And this takes advantage of the fact that on CUDA, were typically memory bound, not compute bound.
So in fact, most of our internal operations happen in floating point precision.
And this is good for precision purposes because if you do all your internal computation in 32-bit floating and only convert to 16-bit floating at the end, you're not gonna have as many, like, you know, sort of catastrophic cancellation or loss of precision events from every intermediate operation in question.
Of course, if you're running enough operations, you might still want to do them in half precision because you might be compute bound in that situation.

So that's most of everything that I wanted to talk about with half precision.
There's one last thing that I wanna put in your brain, which is that reducing the number of significant bits or exponent bits is not the only way to, you know, sort of reduce the amount of memory that your parameters use.
There's another way you can do it, which is you can represent your parameters as integers entirely.
And that's called quantization.
It's another very interesting way to reduce the memory footprint and compute costs of your models.
That's everything I wanted to talk about today.
Talk to you next time.

[^floatingformats]: [Here](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#bfloat16_floating-point_format) are some helpful visualizations on Wikipedia.
[^floats]: For me, the window-and-offset interpretation of floats explained [here](https://fabiensanglard.net/floating_point_visually_explained/) is what made the format intuitive.