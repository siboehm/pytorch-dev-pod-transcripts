---
layout: post
title: "Random Number Generators"
date: 2021-06-20
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Random Number Generators

Hello, everyone, and welcome to the Pytorch Step podcast.
Today, I wanna talk about random number generation in Pytorch.
Random numbers are a very important component of deep learning, You use them when you initialize your weights.
You use them when you use players like dropout.
which will randomly zero out connections.
And in general, the concept of stochastic gradient descent is predicated on this idea that you're gonna sort of randomly you know, process batches in your input data set and, you know, this randomization.
Well, how do you do it? You use the random number generators in PyTorch.
So there's some basic facts about random number generators in any sort of numeric library that Pikers choose to.
And the most important concept is that although you normally in idiomatic usage just say torch dot random and then you just get a vector full of normally distributed random numbers.
which actually is happening under the hood is that there is a random number generator, an explicit generator object, and you're just using the implicit sort of global generator in a situation.
But really, you can create these objects explicitly and use them to sort of separate the random number generation question.
And so when you want to know a bit more about how the random number generator inefficiency is implemented, you want to look at these generator objects which are implemented differently depending on if you're generating numbers on CPU or CUDA, and these contain the important state and the important functions for interacting with this state for the various algorithms that you're gonna use to generate random numbers.
The most important devices on PyTorch are CPU and CUDA, and we use different algorithms for them.
Sorry, so that means that if you, you know, train your model test your model on CPU and then move it to CUDA, you're gonna get different random numbers.
We've talked idly about maybe, like, implementing CUDA's algorithm on CPU, but no one's done it so far.
So in CPU, we just use a good old fashioned Merseni twister r and g.
That's a pretty high quality pseudo random number generator.
It isn't cryptographically secure, but it's fast to run.
A lot of people use it and it has pretty good statistical properties.
On CUDA, we use a different RNG called FILOX.
So Philox is used in CUDA because it has a really interesting property.
Its internal state can be entirely represented as a seed and then an offset into the random number stream that was generated by that seed.
Why is this an interesting property? Well, Mersedi Twister traditionally involves some sort of random number generator state.
And then every time you sample random numbers out of it out of it, this state changes so that, you know, you, like, you move some of the random bits around and that then you do the same thing over and over again.
And so the state is bigger than the seed, which typically is just a sixty four bit integer which means that it's easier to have a higher periodicity that is to say when the VIN number generator starts looping over itself in that situation.
So Philox doesn't need to have some state that you're gonna, you know, put round numbers in.
Instead, it will just calculate the c state right off the bat when you start your CUDA kernel based off the seed and the offset.
and this is important because it means that we don't have to persistently keep a cuda Tensor around representing the R and J state of a Philox Tensor.
Instead, because the seat and the offset are totally are very small, they're just, you know, a single sixty four bit integers, we can send them every time we do a CUDA launch directly using the, you know, scratch space that kudos to colonels allow for sending small amounts of data directly to the colonels without having to do a device, a host of device copy.
So what happens when you use a Felox RNG? Well, we first query the generator object representing a CUDA Kuda RNG, we get out the seed and the offset.
The office tells us how far along we've gone in the random numbers state.
We send these via our CUDA kernel launch to CUDA.
You use CUDA and init to initialize a local scratch space.
So, okay, highlight.
There is a scratch space, but you just reinitialize it from scratch.
And this is okay because what's gonna happen right after that is CUDA is gonna like use it over and over again because you're gonna do something like fill a entire random number entire buffer full of random numbers.
So you can amortize the cost of this state initialization.
And then back on the host side, the host is supposed to statically know how many random numbers your algorithm is gonna use.
And this is usually not too hard to figure out.
Like, for example, if you're, you know, filling a random vector full of random numbers, the amount of random numbers you're gonna use is exactly the length of that vector times, you know, however many random numbers it takes to generate a single element.
So you increment the offset by however many random numbers you would have used.
And so the next time you launch a kernel, you'll start at the next part of the rhino number generation stream, and you don't have to worry about, you know, reusing old numbers in the old case.
There's also some fancy stuff for handling CUDA graphs.
There's a bit of a digression, but I just wanna put it out there, which is that CUDA graphs, which are a way of recording a bunch of CUDA kernel launches and then launching them directly without having to pay for kernel launch costs or any of the, you know, sort of code that Pytorch has to run to actually get to the kudal kernel launch, those hard code are the parameters that you launch kernels with.
And so what that means is that the seed and the offset are traditionally hard coded into the kernel launches.
And so if you wanna then rerun these kernels later via CUDA graph, you would replay exactly the same random number generators.
So there's a little trick that we do, which is we when you're doing pseudo graphs, there's an extra bit of CUDA memory that we do to add an extra offset that you can use to basically program your, you know, CUDA graph fixed seat and offset otherwise to go to some other offset because you wanna run your code again, but with different random number generators the next time.
Okay.
Digression over.
At some point, I'm gonna do a podcast about CUDA Graph Support in Pytorch, but this is not that podcast.
So I'm so we have generators.
We have a CPU generator.
We have a CUDA generator.
These generators use the, you know, simple idiom that tensor and storage also use, and you may notice that CPU state and CUDA state are pretty different.
So in fact, there's two different generator classes and, you know, they they inherit from a common interface, but this interface doesn't actually have a virtual method for getting random numbers.
And if you think about it, this makes sense.
Because, well, you know, like, what good is a virtual method that like directs you between CPU or CUDA when, like, on CUDA, you can't even call virtual methods.
Like, that's just not a thing you want to do in CUDA.
So Like, although, like, standard object oriented design would say, oh, yeah.
You know, you want some method that can get you a different random number depending on what generator you're using.
In reality, what you need to do is you need to refine the type.
You need to figure out which kind of generator you have at the very beginning of your kernel.
So you cast a generator into a CPU generator or a CUDA generator and then just directly access the fields based on what you need.
And so that's how most of our kernels are written.
Right? So you hit the kernel.
You have this type of raised generator.
You figure out what generator it is.
Now you have a, you know, more specific CPU generator and then you use the fields directly.
One random side note.
Our random node generators do have locks on them and We never really agreed whether or not Pytorch's generators or thread safe or not.
Historically, we did protect them with a mutex.
This is like back to the TH days.
So they've kept the mutex as time has gone along.
a one common anti pattern which you should be careful about is the mutex is just protecting the r and g state.
So if you're like doing something like fill ox, You don't actually need to hold on to the r and g lock for the entirety of your CUDA kernel launch.
you just need to take out the lock and then update the offset and then you don't need the lock anymore.
So, you know, try not to like lock the entire things.
Right? The lock is just for accessing the internal state.
But at some point, we should probably figure out how to get rid of the locks because they're not really adding much.
You you probably should deal with locking concurrent access to a generator yourself if you're sharing a generator across multiple threads.
In Python, this is hard to do because, you know, there's a global interpreter lock.
you're usually not running a multiple threads anyway.
And that's most of the important stuff about the generator state in Pytrux.
Right? There's these generator classes.
They contain the state necessary for generating random numbers.
And then various criminals use that state to actually, you know, run the algorithms and output, you know, random floats or run and doubles or whatever it is that you need to do.
There's some interesting stuff also on the front end, which is how to generate random numbers given a, you know, like sort of uniform set of random bits.
Right? Like, for example, if you wanna generate a random double, you can't just take a, you know, random integer and then cast it into a floating point bit pattern directly because that would just be totally non uniform.
Right? Because, like, most of doubles a bit spaces taken up and coding nans.
So you'd get nans most of the time.
So there's like a bunch of algorithms for doing this sort of thing.
and not not really gonna really tell you about all of them.
You can, like, read through the source code and, like, check them out for yourself.
They're they're actually pretty short and they have cool names and like you can read the Wikipedia articles about how these things go.
There is one thing that is kind of interesting that I do wanna point out And that's when we wanna generate normally distributed values, so like your good old fashioned torch dot rand n, we use this thing called the box molar transform.
The way the box mode transform works is that you sample two uniform doubles between zero and one and then you sort of look at what the the sort of angle and the length of the vector pointed by these things are, and you can use that to get out the you can use this to get out the normal normally distributed samples.
But the thing is that to do one of these box millimeter samples, you have to first sample two doubles and you get out to do doubles.
And that's a little awkward if you, you know, only wanted one normally distributed double.
So the way that this works is actually most of our r nGs have an extra little bit of state, which is a a cash normally distributed value.
And so if v because it's like, okay.
Well, I got these two random numbers, but I only need one of them.
The next time I ask for it, I'll give you that.
Instead of having to, like, sample two doubles to produce only one, that would be bad.
And, you know, you wanna reduce the amount of r and g you choose through in this case.
that's, like, that's why there's, you know, these next normal fields on the generator state.
It's for dealing with normal numbers.
And normally distribute the numbers.
And, you know, normal distribution is really important.
So, like, it's worth special casing this kind of situation.
Another thing that is kind of interesting about, you know, like transforming these random numbers is that the boundary conditions can be pretty nutty.
Like, you know, people actually care when you're sampling a floating point number if you're zero inclusive or exclusive and if you're one inclusive or exclusive.
And this is, like, because, like, dividing by zero is pretty bad.
And, like, yes, maybe this only happens, you know, one in every two to the thirty two times, but, like, yeah, that's bad.
And we've had a bunch of, like, very nasty bugs where, like, if you, like, run the thing, like, forty million iterations, like, once upon a time, it gives you an impossible value.
And, you know, over time, we've fixed a bunch of these.
So that's another thing that, like, you have to be careful about when you're working on random numbers.
Okay.
So that's most of everything I wanted to say about random numbers.
This one last thing I wanted to say, which is sometimes you want to build what if you want to, like, take your own r and g and then sort of re implement all of the functionality in PyTorch on top of it? Like, you know, basically plug in your new, like, cryptographically secure r and g instead of Mersenne Twister, and then, like, get out normal numbers and, you know, exponential distributions and all that stuff.
This is something that Pavel Belovitch needed to do for CSP and RG, which was specifically for cryptographically secure random numbers for some of the crypto projects that are going on on top of A10.
And they so this is kind of tricky.
Right? Because As I said earlier, there's no virtual interface for getting numbers.
If there was a virtual interface for getting numbers and the performance was acceptable, you could just, you know, virtualize the generator object and then swap out your own generator object whenever you want it like a c s p r g or just, you know, wanna do something with size or semi twister.
but we can't do that because that's too slow.
We need direct access to the generator state when we're doing one of these vectorized things because we're doing it in the fast loop and, you know, we need everything to inline in that situation.
So what's actually happened is all of our transforms, our random number transforms, our templates And so once you define your custom r and g, you instantiate all the templates for your r and g, and then that ensures that everything gets in line and you get a fast implementation in this situation.
And so all you need to do is just make sure your generator has a, you know, distinct dispatch key, and so we'll make sure you we'll dispatch to your particular you know, random number algorithms instead of anything else.
That's pretty nifty use case of the dispatcher.
Sebastian and I used to argue a lot about whether or not generators should have dispatch keys or not, but mine was pretty nice.
So I like it personally, at least.
Okay, that's everything I wanted to tell you about RNG's.
Talk to you next time.
.
