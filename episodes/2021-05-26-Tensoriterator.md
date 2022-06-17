---
layout: post
title: "Tensoriterator"
date: 2021-05-26
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Tensoriterator

Hello, everyone, and welcome to the Pytruggs dev podcast.
Today, I want to talk about Tensor iterator, but I'm going to go about it in a sort of unusual way, which is imagine you're walking into a software engineering interview and you're wondering what question you're gonna be today, and your interview assistant says, okay, Edward.
Please add two vectors together.
And I say come again? Yes.
I given two vectors a and b, add them together, giving a new vector which contains the pairwise sum of each element in each of the vectors.
And I think to myself, oh, this how complicated could it be.
Right? Right of for loop iterating over the size of the vectors and, you know, just look at the two entries, add it together, and then, you know, set it in my output and return the result.
You know, easy.
Are we done? And then the interviewer gives you an evil smile and they says, okay.
Now, what if you want to make it really fast? and you want to make it work in a lot of situations.
And so this situation is a sort of exactly the situation that Tensor iterator is in.
Right? On its face, the job that tensor iterator is trying to solve is very simple.
Namely, given two tensors, you know, do some point wise operation on them.
So, you know, given two tensors, you know, like look at the first element.
And the first one, look at the first element on the second one, add them together, that's the first element of your new tester, and keep doing it step by step by step to the step.
And you might think, hey, this should be really simple.
And it is sort of, but it turns out that when you're in a library like PyTorch, there's actually a lot of different conditions and also a lot of different performance optimizations that you might want to apply in this situation that, like, actually end up making Tensor Editor very, very complicated.
So the goal of today's podcast is to talk a little bit about all of these things that, you know, go into making a tensoriter to do work.
So where to start? So let's start from the very beginning.
So I gave you these two vectors and I wanted you to add them together.
And one of the first things you just asked me is, well, okay, what are their sizes? Are their sizes the same? because, you know, a tensor is not just a one dimensional array.
It's actually a multidimensional array, right, possibly with arbitrary dimensionality.
And so, you know, when you wanna add two things together, it turns out that, you know, adding two things together, doesn't require the two input sensors to be the same shape.
In fact, Pytorch implements something called Broadcasting.
We got broadcasting from NumPy, which also does it.
What broadcasting says is it basically simplifies the situation when you have a tensor and you wanna like add a single scaler to it.
So if I have a tensor and I wanna add two to it.
That just means, hey, add two to every single element in the tensor.
And this is a special case of broadcasting.
Broadcasting actually sort of generalizes to arbitrary dimensions.
So let's say that I have a five by four by three tensor and I've got a sort of sample tensor of size three that I wanna add for every single element in my five or by four by three.
Namely, I wanna do it five by four times.
Right? twenty times, stamp in this extra three, and replicate it in all of the situations that it shows up.
And broadcasting also supports that.
The way you figure out how it tends to broadcast, by the way, is you sort of line up their sizes but to the right rather than to the left, and then just pat it out so that it goes all the way to the end.
And so their sizes from the right have to match up and then everything else gets replicated.
Alright.
So one of the first things you have to do when you're doing sensor iterator is when you wanna add things together is actually, like, we will accept inputs that don't have the same sizes, and you need to do something reasonable in that in that case.
Namely, you need a broadcast in a situation.
Okay.
Sure.
So let's say that you know how to do broadcasting and you've written the algorithm to figure out what the output shape should be.
You know, what else is there? Well, I didn't tell you what the, you know, types of your inputs were.
Right? And, you know, normally, if I give you two vectors in an interview, you'd just assume they have the same type.
But what do they don't have the same type? Well, in Pytorch, we have this thing called type promotion, also taken from NumPy, which says, hey, under some situations, we are willing to add together tensors, which have different types.
Once again, why is this desirable? Well, sometimes, you know, you have a floating tester and you've got an in tester and you just wanna add them together.
Right? You wanna treat the integers as if they were floating point numbers and let's do this addition.
So there's a table and I'm not gonna tell you this table in the podcast, but there's a table which imagine that, you know, you have all the different d types in PyTorch, like n thirty two, n sixty four, n sixteen, float, double, etcetera, etcetera.
and then you got another access on the table, which is all the d types as well.
And the tie promotion table tells you, given two inputs of these two d types, what is the output? d type in question.
And so this is something else Tensorensorensorator has to, you know, compute.
Right? Which is that, hey, you know, what is the actual output d type? because maybe the input types of my values are not the same.
Oh, but it gets better than that because hey, you can also give a addition operation explicit out tester that you wanna write the results to.
Does the out tester have to match the computed d type in this situation.
The answer is no.
It doesn't.
It can be different.
And yeah, we're also gonna promote as necessary to, like, fit the output into the output d type you give us.
So, hey.
Alright.
Like yeah.
So, you know, you asked about what the ties could be.
We said they could be anything.
We asked about what the shapes could be.
They could be anything.
Does it get worse? Yes.
It does get worse.
Okay.
So I mentioned that addition can take an optional out tester.
Right? And so what does this mean? It just means that, hey, when I add these two tenders together, Don't allocate a new output buffer for the situation, just write it in place into this preexisting buffer.
what happens if that output buffer aliases with one of the inputs? And this is like actually kind of tricky to deal with.
In general, like, the sort of aliasing situations make, you know, otherwise straightforward algorithms a bit more complicated.
So in some situations, it's okay for this alias thing to happen.
Right? So, like, let's imagine that I am adding a tensor in place.
Right? So I've got this tensor I wanna add two to every element in it.
What actually happens in this situation is I put in the tensor as an input, and I also put the tensor in as an output.
And because addition is sort of atomic.
Right? Like, I just read out from the memory and then I do my operation and then I write out back to the memory without ever, like, looking at any of the other memory locations, this is fine.
And, like, nothing bad will happen if the inputs and outputs alias with each other.
But let's imagine that my output sensor actually is started in a funny way.
For example, what can happen with strides in Pytorch is that multiple logical locations on the tester can refer to the same physical memory.
Right? We talked about broadcasting while broadcasting exact is exactly a situation like that.
So what happens if, you know, you've got multiple logical positions pointing to the same physical location? Well, let's say you're processing, you're inputs one by one, and so okay, I wanna add two to some location.
So you go ahead and you read out the physical location, do the addition, and write it back out, uh-oh, the next time you read out from that physical location again, because once again, this is one of those tenses where multiple logical positions mapped to the same physical position? Well, you've already clobbered the old value there.
And, well, sucks to be you, you just are gonna get total garbage in this situation.
So something else Tensor Editor has to do is it needs to make sure that there aren't any sort of illegal overlaps between the inputs and the outputs.
And also sometimes, you know, with the Also, the output itself needs to make sure that it doesn't overlap with itself, which can also cause problems where, you know, you write to the output location and then you write to that output vacation again, clobbering the original value.
Oh, man.
By the way, the problem of determining whether or not there is an overlap is actually, like, equivalent to, like, solving diafentine equations.
So Pytorch does an approximation.
It would be really really difficult to do this properly.
Oh, one more thing.
This this like alias thing where the destination and the source could overlap.
This is one of the reasons why, like, there's a difference between MEMC copy and MEMC move in the CAPI.
Right? One of them is allows for aliasing and another one doesn't.
And you have to be careful when you're writing code to figure out whether or not Aleson can occur or not.
And since Pietro just a library and it gets called by anyone, we have to basically assume that arbitrary aliasing can happen to anyone.
Alright.
So we talk about shapes We talked about d types.
We talked about memory overlap.
Are we done? No.
So I mentioned about strides.
Right? So we talked about housestrides can be used to, like, implement broadcasting.
And so what do I mean by that? Well, you know, in PyTorch, we support this operation called expand.
And so what expand does is it takes a tensor of some size and then expands it to be some bigger size with the same element, you know, repeated repeatedly.
But we don't actually materialize all of this in memory.
What actually happens is, you know, it just gets stamped out in duplicate copies.
And the mechanism by which this happens is a stride.
The stride says, you know, once you advance in some dimension, your index in some dimension, how should the, you know, physical location change? Right? And so normally in a, like, contiguous dense denser, if I advance and I dimension, that means I should skip ahead however many, you know, a hundred bytes, four hundred bytes, whatever to get to the next sort of chunk of data in this case.
But when I broadcast, I just say, oh, that number is zero.
So I'm not going to advance at all.
So broadcasting is a degenerate case of striding, but in fact, striding has a lot of other possibilities.
Right? And, you know, what it comes down to is that when I have this, like, flat contiguous piece of memory, there are multiple ways I can interpret it as a multi dimensional tensor.
And the, like, very, like, simplest example of this is when I think about how two d matrices are, represented.
Right? There's this concept of row major and column major major c's.
Right? When you read out the numbers, you know, one, two, three, four, five in physical memory, Does that correspond to reading out a column or a row? And Pydge supports both of these simply by just specifying what the strides of a tensor are.
Okay.
So you can't assume that, like, the layout for both of your sensors is the same.
And so, uh-oh, another thing sensor iterator has to do is given the two inputs what should my output layout be.
Right? Because, you know, if I give you a column major enhancer and a row major enhancer, well, I had to make some decision about what the output should be.
This is there's a very complicated resolution algorithm for it.
But, like, one of the properties that it wants is if, like, I add a column major tensor to a column major tensor I still get a column major sensor.
And similarly, if I add a row major sensor to a row major sensor, I also get a row major sensor.
Is that it for strides? Not quite.
Okay.
So there's something else that happens.
So we're getting out of the realm of correctness.
Right? We're Like, we just needed, like, deal with all these things, like shapes and d types and layout because they're part of the public facing specification.
And we're now getting into the how do we actually make the algorithm run fast? So one of the things that is with strides that, like, makes them kinda slow is, like, if you have a really big dimensional tensor, Well, you have a lot of strides.
And if you wanna index, you know, the indexing formula with starting is, you know, take the first index multiplied by the first stride and added to the second index, multiplied by the second stride and so forth and so forth and so forth and so forth.
So you can imagine with a really high dimensional tensor indexing computation actually takes a lot of time.
And in fact, we try very, very hard not to do arbitrary dimensionally indexing and most of our helpers we're doing indexing require us to know exactly what dimensionality a tensor is.
By the way, that's also the reason why, like, say, eigen is actually templated on dimension because it's way easier to generate efficient code in this situation.
But tensor iterator is supposed to work on tensors of arbitrary dimensionality So, like, how do we do this efficiently? Well, another important optimization that we do is sometimes we have multiple dimensions, but they're actually all contiguous.
Right? Like, let's imagine that I have a contiguous tensor, a contiguous, you know, five dimensional tensor, and it's just laid out in memory, you know, exactly one to three for five six seven eight nine And I'm adding it to another five d tensor.
Well, I don't actually care about the dimensionality in this case.
Right? Like, if the dimensions are exactly the same, there's no broadcasting, there's nothing like that.
I could just treat these both as one dimensional tensers and add them together, and that would give me exactly the same result.
Well, okay, I get a one dimensional result and I have to reshape it into a five dimensional tensor.
But like the computation between these two cases are the same.
So another optimization that Tensor iterator needs to do is it needs to say, okay.
Well, you know, I've got this n i n dimensional tensor.
It's got all these strides but what I'm gonna do is I'm gonna coalesce these dimensions where when I have contiguous stripes, I'm just gonna treat them as one mega dimension and so I don't actually have to spend time doing indexing computation inside them.
Oh, man.
So that's a bunch of stuff the sensor reader does.
Okay? So like, you know, we're like looking at sizes, we're looking at strides, we're looking at d types, we're looking at overlap.
Is that it? Well, not quite.
So remember the interview question where Right? Like, so it's like, okay.
How do you add two vectors together? oh, I will just write a loop and I will add the elements together and I'll be done.
And then your interviewer says to you, okay, how do you make it faster? Well, there are a lot of things you can do to make it faster.
So one thing you can do to make it faster is you can parallelize it when there are lots some data.
Right? So, you know, what what does that mean? Well, you know, I've got this giant denser.
I'll just split it up into chunks, into grains.
And I'll ship each chunk to some thread.
And the threads will, you know, do the addition in parallel on each of them.
And, you know, like, if I I'm not trying to run-in a multi threaded environment.
I get all the quirks to myself.
You know, that's a big speed up because, well, one, you know, the data can be shipped out in this way without, like, too much interference.
And two, because I've got all these cores and they all have ALUs and they can actually be usually doing computation in this case.
So there's, you know, when you run a CPU kernel, alpha tester iterator, parallelization is something you get for free in a situation.
But wait, there is more So you're doing your addition on a single thread.
Right? And it's like, hey, you know, please add the first element, please add the second element, please add the third element.
Well, we can do better than that.
Right? Because there's this little thing called vectorization.
See, my earlier podcast.
Vectorization means I can actually do chunks of, you know, multiple numbers at a time and take advantage of that AVX Silicon in my CPU.
So, you know, not only am I going to paralyze, but I'm also gonna vectorize what I'm actually doing the cyber site elements.
And that's also something Tensor Editor takes care of.
Okay.
So I paralyze my code, I vectorize my code, can it go any faster? Well, yeah, it can.
Right? Because, you know, the whole point of running things in PyTorch is GPU acceleration.
Right? GPUs are these massively, massively parallel processors and, you know, they have way more parallel than just the poor vector units on our multi core CPU can be.
So another thing generator does is it lets you write kernels that work both on CPU and on CUDA while sort of saving all of the, you know, other stuff like sheep and d type and, you know, layout that common stuff, letting you just do that once.
for the two implementations.
And that in a nutshell is, you know, most of the things TensorFlow does.
There's more stuff that I haven't really talked about and glossed over.
But at a high level, Tensor iterator is, you know, sort of pulling its weight in two ways.
One is that it is doing all of the complicated semantics for point wise operations that you just don't think about.
But, like, are these things that, like, people rely on working uniformly for all binary operations.
And two, is it makes it to write reasonably efficient code when you're writing things in PyTorch.
And it does so without, like, needing a just in time compiler or anything like that.
Right? It like gets compiled once.
It doesn't take up too many much binary size.
and you get decently fast kernels that work in a huge variety of situations.
Not everything is great with tensor iterator.
Tensor iterator is kind of slow.
It does a lot of, you know, bookkeeping and that bookkeeping adds up.
We'd like to make it faster, but this is one of the reasons why it's been so hard to replace because it really is doing a lot and, you know, ask what you can do for TensorATOR, I say.
That's all I wanted to say for today.
Talk to you next time.
.
