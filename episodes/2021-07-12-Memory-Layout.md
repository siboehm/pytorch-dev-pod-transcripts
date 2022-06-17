---
layout: post
title: "Memory Layout"
date: 2021-07-12
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Memory Layout

Hello, everyone, and welcome to the Pyturbine Podcast.
Today, I want to talk about memory format in Pyturbine.
To answer what memory format is.
I want to talk a little bit about how tensers are laid out in memory.
Tensors are multi dimensional qualities.
You can have as many dimensions as you want in a tensor, and that distinguishes them from like good old fashioned vectors, which you know, are one dimensional or even matrices which are two dimensional.
And so, you know, although there's this dimensionality, although, like, you know, you can have like, cubicle data or like this arbitrarily high dimensional data.
When it comes down to it, when you wanna actually store this data on your in your memory, Well, memory is linear.
Right? Like, standard CPUs let you address memory via a a, you know, numeric pointer that, you know, is laid out entirely, you know, linearly.
Right? Like you've got address zero, then you've got one, two, three, and so forth, and so forth.
So when you have a multidimensional tensively, you need a way to linearize this into some actual concrete ordering in memory that, you know, doesn't have any constant intervention.
It's strictly one dimensional.
and this linearization is the layout of a tensor in question.
To give an example, it's helpful to look at the two dimensional case.
So in a two men dimensional matrix, let's like imagine, for example, a matrix where reading left to right top down, I have one, two, new line three, four.
So this is a square matrix, one, two, three, four.
I want to figure out how to lay this out in memory.
And there are actually two reasonable ways you can go about doing this.
So one is you can read out the rows and sort of paste out the rows side by side in memory.
So when I so I read this to you left to right top down.
Right? And so I said one two and then three four after the new line.
And so you can actually just lay it out in this order.
So, in memory, you see one, two, three, four.
like directly lay out in this way.
And this is what we call c order or row major order because you first do the rows.
and this is what like the layout you'll get with Pytorch and like with any sort of c programming language where you do a multidimensional array, this is exactly how it's gonna go.
but there's another choice.
Right? Instead of reading from left to right top down, I could read top down first and then left to right.
I could do the columns first.
And this gives you so called column major layout.
So when laid out, I would have one three two four.
Right? Because the first column is one three and the second column is two four.
And so in this case, like, the layout on disc is different.
And in fact, if you were writing in four tran, this is in fact the order that your arrays would be.
Which one is better? Well, it depends.
I mean, it depends on what kind of algorithm you wanna do, and Like, either of these could be valid representations for your data in question.
Of course, it's often there is often a convention because when people write kernels, they usually want to make an assumption about how things are laid out.
And so for example, in Pytourch, the conventionally is you would just assume things are real raw major unless, you know, like, something specialist happened.
Another example of LayOut, and this one is much more germane to deep learning, is an image process So an images, what is the typical thing that you need to do? So an image consists of a bunch of pixels of some height and some width.
and typically images have multiple colors.
So you need a channel dimension that represents, you know, this is the red color, this is the green color, this is the blue color.
And of course, because you're typically doing, you know, gradient descent on batches.
You also have a batch dimension so that you have a bunch of images sort of stacked up on top of each other.
And so the standard representation for an image in PyTorch is what we call NCHW.
So what that means is first, the first dimension is the batch dimension.
The second dimension is the channel dimension, the third dimension is the height dimension, and the fourth dimension is the width dimension.
If you imagine if you wanna imagine what this looks like in memory, for a moment, let's forget the batch dimension.
The child dimension comes first.
And when the child dimension comes first or is the so called outermost dimension, that's the thing that changes least frequently when you are going through the actual linearization in order.
So just to like, you know, go back to the example.
Right? So you're gonna have a red and a green and a blue channel.
So what is the image gonna look like in memory if you have a CHW and CHW layout, well, first, you're gonna see all the reds.
r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r r, like all the red pixel values.
you know, for each row going going down on your tensor.
And then you're gonna see all the green.
right, for once again the rows are going down the image.
And then you're gonna see the blues.
Right? So what you can imagine in memory is you've got this region for red, region for green, and region for blue.
And why is it like that? Well, that's because the channel is first.
So it is the thing that changes least frequently.
Of course, there's another layout that is commonly used and that layout is called NHWC.
So instead of channels being first, channels are last.
And so in this particular case, right, because channels are the innermost dimension, it's the the thing that changes the most frequently.
So in fact, if I looked in memory, what I'd see is RGB, RGB, RGB.
So, like, you know, every time I'm, you know, handling a pixel, I'm gonna put down all the values for each of the channels if we're moving onto the next pixel.
So it looks like, you know, an actual, like, what an actual, like, you know, LED screen would actually look like in a situation.
And of course, we sure these are better.
Well, once again, it comes down to the algorithms.
but it turns out that in, you know, some convolutional algorithms, they're just more efficiently implemented in NHWC.
So that's like sort of what a lot of people want to be able to use channels last layout to make their code run faster because of these special kernels.
So we could just stop here and say, okay.
Well, Ed, you know, I Great.
I know what n c h w is and I know what n WC is.
So now, you know, whenever I get an image sensor, I just need to know if it's one of these or the other, and aren't I done? And that is okay.
But, you know, when we wanted to add support for both of these tensor types, we had a problem.
And the problem we had was we didn't want to actually force people to keep track of, you know, what layout their tenses were.
Right? Like, they could do this.
and they were doing this.
But it was pain in the ass to actually have to deal with this for all of our operators.
Like, just think about convolution for a second.
Right? Consolution needs to know what we're actually going to, you know, do the convolution over with regards to the channels and what we're not going to do over if you have an NCHW temperature, you need the convolution to operate with the channels in the first position.
And if you have an a h w c tester, you need to have the convolution operated with channels in the last dimension.
And these are different algorithms and you need to actually tell convolution what type of tester you actually pass in? And tensors are these very dumb, you know, like n dimensional arrays.
They don't actually have any semantic content.
So that's something you'd have to keep track of externally from the tensor.
And that's a pain and we didn't want to have to do that.
So what did we do? Well, to answer this question, I have to take another detour and talk a little bit about how we implement this linearization under the hood in PyTorch.
and this is done using strides.
So what are strides? So I said that, you know, layout, memory layout of a tensor of an n dimensional tensor It's all about taking your, you know, various elements and then laying them out in a linear sequence of addresses in memory.
While strides are a way of computing given any given coordinate in the logical tensor, where does it physically lay in the actual linear memory address layout.
So let's just talk a little bit about, for example, c layout, which is what Pytorch does.
So in c layout, right, the outermost dimension, the dimension that comes first is the one that changes least frequently or in other words, like, to get to the next element, the next slice in that dimension, you have to jump a bunch of elements further.
Right? That was the r r r g g g g b p v.
Right? So you need to, like, jump for r's to get to the g's and then another four to get to the b's.
But on the other hand, if you're in innermost dimension, one on the very end, Well, you just, you know, can look at the next element and see what the element is in that situation.
And this is the concept that Strides do.
So Stride says For any given dimension position, how much do I have to advance the physical memory pointer to get to the next element corresponding to that dimension.
So if your innermost dimension is fast moving, then those sorry.
If the inner mass dimension is the one that changes, you know, all the time contiguously, then I say strive for that as one.
Because if I like want to move to the next element, I just go to the next physical Mary layout.
It's all laid out contiguously.
Whereas if I'm on the outermost dimension and I want to, you know, jump really far, then I might give it a stride of say four if this was you know, size four tensor in the outer dimension.
And that just means, hey, to get to the next element, you have to jump four elements ahead.
So going back to our, like, original example, one, two, three, four, that square matrix.
Right? The strides for this in sea layout would be two, one.
To get to the next element in a row, you only need to look at the next next spot in your contiguous memory.
But to get to the next sorry.
To get to the next value in the row, you just subject to the because the next get to the next value in the column, to get to, you know, to move the row down, you have to jump past the entirety of the row.
And so that's why the straight is two because the two is the size of the row that you have to skip across to get to the next thing.
And of course, if you have four tran layout, then your strides are simply one three because when you want to see what the next column is.
Well, the columns are now laid out continuously, so you just advance it by one.
But if you want to see the next row, well, the those are not set out contiguous, and you have to jump.
And so the stride in that cases too.
Right? So seal out is two one, but the strides are in decreasing order.
And fortunately, layout is one two.
The strides are in increasing layout.
And in fact, you can flip between these two strides just by using transpose and PyTorch, which doesn't do a copy.
It just, you know, fiddles around with the strides and then gives you a new tester with those different strides.
Okay.
So what the heck does this have to do with memory layout? Well, we had a very clever idea to make memory layout work.
So Pie George originally only supported n c h w and all of our convolution operations assumed that you would put the channels first when you call them.
So what we said is hey, let's just double down on that.
So the user visible API, the logical view on Tensors, always requires channels to be in the first position right after batch.
But if you want to use channel's last layout, Well, no one said that n h the n c h w logical layout had to correspond to n c h w physical layout.
Right? It could.
And that would be the case when the strides are strictly decreasing.
But it could also remap to a physical layout that actually holds things out and NHWC.
I'm not gonna tell you what the strides are in this case because it's not the obvious one.
It's not the permutation from n CHW to nHWC.
It's the reverse permutation because reasons.
Try deriving that by yourself if you're actually interested.
And so by doing it this way, right, like the physical memory layout is what the kernel actually cares about because, like, the kernel, like, is going to run faster because of something that it's doing regarding memory locality.
But at the same time, we can still get the same user experience where, like, a convolution always takes in a n c h w tensor.
and this might happen to be one of these weird transposed sensors that is represented differently in physical memory.
Some things to know about internally how we implement this.
So although we store strides and in principle you can calculate whether or not something is n c h w or n h w c from the strides.
It's kind of expensive to do this.
So we actually have this giant bit filled on tensive.
that, like, has all the common memory layouts that you want to often test for, like, when you're doing convolution.
And these are all just pre computed based off the strides.
to make access fast.
I kind of hate this design, but it is very expedient and it indeed does have performance benefits.
There's one last interesting thing about memory layouts done in this way that I wanna tell you about, and this is the ambiguity problem.
Let's imagine that I have a one by one tensor Well, the strides for this tensor are one one.
Why is it one for rows? Well, because there are no rows.
And even if there were rows, I would only have to go one to go to them because the size of the row is one.
So, like, you know, advancing it is easy.
What I have strives, there are like one one where I have one of these one size dimensions.
I cannot tell what the layout is.
I cannot tell is a real major or column major because the strides just don't have any information for me.
And this is problem because one of the things that we need to do when we are doing memory layouts is we need to propagate memory layouts.
Right? Like, it's no good if I feed in a NHWC Tensor expecting convolution to get it and use my efficient, you know, channels last kernel, if somewhere in the middle, I have an operator that takes in one of these tensors and then just calls contiguous on it.
And the meaning of contiguous is put it in n c h w format.
So it'll go ahead and do that.
And then, well, such to be you, like, you've just lost all the optimization opportunity.
And so when you have tensors which, like, lose this layout information, you might actually make the wrong choice and turn it back into an NCHW tester.
if you, like, expand the size.
This has happened.
We, Natalia, Kimo Schein, fixed a bunch of these cases when we were originally trying to figure out how to do this.
And, like, most of the time, the way we resolved it was, like, there was some extra data, there was some other tensor that we could rely on to get the information that we needed.
There's also some conventions you can do when you're writing out the strides because actually you have a lot of degrees of freedom when a stride is for a size one or size zero tester.
Right? Like, if your if your tester is only size one, it doesn't matter how big or small your shirt is.
because you're never gonna actually use it.
You only ever multiply it with a zero and, you know, you never multiply it with one because that would imply there were two elements.
I had a proposal for solving this problem called lay out permutations where the idea was instead of only throwing the strides We also store a layout permutation.
This is exactly what the permutation is.
This would also solve the ambiguity problem because when I have strides one one, I would also know via the permutation if it was zero one or one zero.
But we never implemented this because it was kind of a lot of work and we solved most of the most pressing problems by just annually fixing them.
So that said about memory format.
Memory format lets you, you know, move around your dimensions and get faster kernels.
That's everything I wanted to save for today.
Talk to you next time.
.
