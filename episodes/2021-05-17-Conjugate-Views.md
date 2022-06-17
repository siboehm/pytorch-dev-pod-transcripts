---
layout: post
title: "Conjugate Views"
date: 2021-05-17
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Conjugate Views

Hello, everyone, and welcome to the Pritchard dev podcast.
Today, I want to talk about a new feature that is going to be landing to master soon for complex numbers.
namely conjugate views.
To explain what this feature is, I have to backtrack a bit and talk a little bit about complex numbers first.
So what are complex numbers? Complex numbers are a form of numbers where instead of only having a single real number, representing a quantity in question, you have both a real quantity and an imaginary quantity.
And the invariant, right, is that the imaginary quantity, you know, when squared, is it negative? And no no positive or negative real number when squared gives you a negative number.
So that's what makes imaginary numbers different.
This sounds kind of strange.
And, you know, for the longest time, neural networks don't really use complex numbers, but in lots of, say, signal processing applications, you know, complex numbers have a lot of interesting properties that make it actually really good for, you know, doing certain types of computations.
So if you're doing some, like, for example, fast fourier transforms complex numbers arise very naturally.
And there's also a line of research looking into how to use complex numbers for useful things.
Actually, when the complex numbers project started, it was a physicist, Roger Lowe, who sort of came and was like, hey, you know, I think this would actually be really useful.
Took us a while to actually listen to him, but, you know, we we got there in the end.
So when you're doing complex numbers, there's an operation that is really really common and is called conjugation.
So what is conjugation? conjugation says, Okay.
If I have a complex number a plus b i, where i is the, you know, constant that when squared gives you negative one, conjugation is taking this number and giving you back a minus BI.
Now, reasons why conjugation in Consequence numbers are very common is sort of beyond the, you know, scope of this podcast.
But one way to think about it is if you, like, think about, like, your linear algebra class that you took in undergrad.
Okay? If may maybe if you took the theory based one because I don't know they really go into complex numbers on the more practical linear algebra classes.
One of the things you do is you, you know, talk about fields on real numbers.
and, you know, you do a bunch of stuff on them and you learn some properties out learning your algebra.
And then you're like, okay, now you can generalize to complex numbers.
and, you know, you have to, like, change all your definitions to make things work.
And one of the things that happens is, you know, everywhere you were doing transposes in your, you know, old theorems.
Suddenly, you're doing hermicians.
You're doing ad joints.
You're you're basically taking both the transpose and the conjugation of the matrix in question.
Whereas, you know, in the real universe, you were just transposing.
And, you know, you just you just need to do this to make all the rooms work out.
And, you know, if you're really, really curious why this is the case, I recommend, you know, like, taking a theoretical linear algebra class and just sort of spending some time stealing with the Theorems.
Okay.
So conjugation is a really common operation and, you know, it's really simple, right, like you just started doing the negation on one part of the compass number.
And so typically, right, you're doing the conjugation because you're about to do some other operations.
So If you are doing matrix multiply, you know, a common thing to do in standard linear algebra is, you know, matrix multiply a with b transpose.
well, you know, in the complex universe, that's gonna look like something like a matrix multiply with b transpose and conjugated.
And here something very curious happens.
So if you think of conjugation as just a operation where, you know, if to conjugate a tensor, you know, you take your tensor, and you produce a new tensor with, you know, everywhere it was a plus b i.
It's a minus b i.
Then this matrix multiplication version is actually a bit less efficient than its old version.
When you did a, matmole, b, transpose, we didn't actually ever do the transpose.
Because remember, Pytorch supports strides on tensors.
So if we wanna take if we wanna take an operation like TRANSPOSE and do it without actually doing the computation in question, it's actually an o one operation.
You just take your tensor and you swap the strides.
So instead of saying, okay, when you move in the y position, like say that that you're indexing x y, only move one.
Instead, changing the y position means moving entire, you know, row.
And moving position is what, you know, you only move one on.
And by, like, simply switching the stride so that instead of, you know, moving one, you move a lot in the y case.
you are representing a transposed sensor.
And actually, so if you've got a back end implementation of matrix multiply, and knows how to implicitly do transposes.
For example, blosses, you know, matrix multiply has a fly that, you know, lets you specify if they're, you know, our argument is transposed or not, then you actually can just avoid having done the transition at all because you just, you know, say, okay.
Well, I wanna do a transposed matrix multiply.
where the right argument is transposed, and you can just call the kernel directly and you're all good.
And we never actually do the transposition.
And transposition is kind of expensive.
you gotta allocate memory for it blah blah blah.
So you don't really wanna do that.
Okay.
But conjugation.
Right? Well, conjugation is weird.
because, you know, conjugation actually involves negating, you know, half of the numbers in your tensor.
And so strides don't really work for this.
And so you're in this weird situation where, oh, well, it likes to be me.
I have to conjugate the tensor and actually, you know, create a new tensor.
And then I can I guess I can do the transpose tricks and then call my, you know, a complex blast, matrix one play implementation.
But this is a waste because, actually, blast provides a fused matrix multiply with a transposed and conjugate on the second argument.
And so, like, yeah, that's faster because, you know, it's just faster to have the Fuze operation.
It's why people like using the jet fuser.
Right? Like, you're often memory bound in these situations and, you know, being able to do this fusion is very profitable.
So what's a poor person to do? So we haven't been hot a bit, and, you know, we talked to some of the experts on, you know, basically doing complex numbers with neural networks, namely Bodecker.
And, you know, we we talked about a few options.
Right? Like, one option was, okay.
Well, we're just gonna you know, provide a new matrix multiply that, you know, explicitly takes a little keyword argument that says, okay, do you wanna conjugate the output? That looks really ugly.
Right? Like, if you're just writing some math down in Pytorch, you wanna just say x, you know, at sign, y dot h, and you want that to work.
Right? You want you want to be able to write code that looks like math.
Like, yes, in principle, we can, you know, write lots and lots of fuse operations and tell people to, you know, look up, you know, some fuse operation for whatever operation they wanna do.
But they don't wanna do that.
Right? They just wanna write math And then hopefully, you know, some compiler or something some smarts in your program are good enough to actually, you know, run that efficiently in that situation.
So we really wanna be able to write, like, this operation and actually have it diffused in the situation.
And so the next thing you tend to think about in this situation is, okay, maybe we can do some sort of lazy tester.
Right? So I've talked about lazy tenses a little bit in the past in this podcast.
But once again, what's a lazy tester? A lazy tester is like, you don't do the operation immediately.
Right? You just wait and see if you run some other operation, and then if it's profitable to, like, fuse in that situation, well, good for you.
You were lazy.
You didn't do the original operation.
so now you can do the fuse operation.
But lazy done's is a little difficult to implement.
And one of the things that makes them difficult to implement is that laziness means that operations which are ordinarily reads can turn into rights.
What do I mean by that? Well, lazy evaluation, you know, as popularized by, say, Haskell, the functional programming language means that you guarantee that you only do the operation once.
So say you have a tensor and you request it, you lazily conjugate it and then you request the value of the conjugation and no fusion is possible.
Under a lazy scheme, you're obligated to actually, at this at the point in time, you do the read to actually materialize the conjugate tensor and then go ahead and do this if you wanna do.
And this makes things a little complicated if you want to, you know, be in a multi threaded environment because, okay, while you're doing a write on a read, that means that, you know, you actually have to start synchronizing your reads and that's actually kind of complicated blah blah blah blah.
Okay.
So And also it's kinda different from this transition.
Right? Transposition was really elegant.
You just allocated a new tensor with different strides, and then it just implicitly fused once you call the function question.
Namely, you weren't doing lazy evaluation.
You were doing call by name evaluation.
where you were willing to, you know, do the transition at every new site of the transposed sensor if necessary.
But, like, in practice, you know, most things get to be fused in this situation.
That's not that's not entirely true.
Like, a lot of operators in PyTorch don't support noncommuters app, but transpose sensor doesn't count as contiguous output.
They're around contiguous on it.
Right? They'll transpose it on the spot when they need it.
But this is a good trade off for us because most of the time, you know, a fusion is actually possible in the situation, or, you know, it just doesn't really matter, you know, because you know, you're only using the transverse sensor once.
So whatever, like, you know, delaying it for later with possibility of duplication is fine.
So we want something that works kinda like transpose, but for conjugation.
And so conjugate views are a way to make this work.
Okay.
So how does it work? Well, you've got your tensor and you want to make a new tensor.
o one.
So you wanna share storage.
You can't copy storage because then it wouldn't be constant time anymore.
And you want it to somehow represent having done the conjugation.
So I'm gonna cheat and then just say, okay, we're gonna define another bit field on Tensor that says whether or not you should interpret the storage as needing a conjugation or not.
So if you have a normal tensor where in memory you have three and then four i, and the tensor doesn't have the conjugate bit, then this entry represents three plus four i.
But if you do have the conjugate bit set, Even though the physical memory says it's three plus four i, you actually interpret it as three minus four i.
So okay.
So we've got our o one tensor allocation.
Right? You just allocate a new tensor, share storage so that the conjugate bit to be one.
Now what? well, you're done.
That's it.
Okay.
It's it's not as easy as that.
So if assuming that every operator knows how to respect the conjugate bit, right? Like, basically, like, if you look at the tensor, you need to look at the conjugate bit.
It's it says, oh, if you need to, you know, interpret the code differently.
Assuming that you have all the operators working this way, Then you have, you know, a o one hermeshian operation.
Right? You just allocate a new tensor, you swap the tensor, you swap the strides.
and you set the conjugate bit.
Easy, peasy.
And as long as all the kernels know how to deal with this conjugate bit, everything's great.
well, making everything actually understand the conjugate bit is kinda difficult.
Right? Because we have a lot of operators, you know, seventeen hundred plus.
And, you know, we don't really wanna be editing all of our operators to, like, you know, pass in, okay.
If the input is, you know, conjugate it, then please, you know, conjugate it, like actually materialize the conjugation and do the operation in question.
Blah blah blah blah.
Okay.
So that's kind of difficult.
So what do we do? Well, we have this nifty feature called the back end fallback.
And what a back end fallback does is it lets us say, Okay.
Whenever you see a tensor that has the conjugate bit set, run this special piece of code unless you've told me otherwise.
So it's a fallback because, you know, you can override the behavior of this, but if there's no override, if there's no actual implementation, we call the fallback in this implementation.
and we can use the fallback to implement the okay while, you know, I forgot a kernel.
It doesn't understand how to respect the conjugate bit.
So I just have to get rid of all the conjugate bits before I call the kernel in question.
And the conjugate fallback will make sure we apply this universally to all functions, even custom registered functions.
So, like, what what does this do? Right? Like, so if I've got a functional operation and I want to run a operation that doesn't understand conjugation on it.
Well, let's see.
So, you know, I've got some arguments.
Some of them have the conjugate bit set.
I need to get rid of the conjugate bit.
So I just go ahead and conjugate them producing new inputs that, you know, whose physical memory represents the conjugation.
So there's no extra interpretation that needs to be done.
and then I just go ahead and call it original kernel.
Very easy.
The logic is a little more complicated in the in place case because, you know, you can't just, you know, change the conjugate bit on the tensor.
There's other tensors that may be aliasing with that storage.
you know, the the the conjugation status of it is related to the storage, not the tensor.
So you can't just conjugate a tensor in place by flipping the conjugate bit on the tensor, you need to do something to the storage, namely actually conjugate the storage.
But you can you can make it all work out, and it's a pretty fun exercise to see how to do it.
And then what do you have? Well, you've got conjugate views.
Right? You've got these views of tensors, you know, views in the sense that if you mutate the view or you mutate the base tensor, the views all all other views so the sensor get updated.
So there's got a view.
But it's not a view in the traditional sense.
It's not a view in just striding or just, you know, swizzling around the data.
It's actually a view in terms of some transformation on the data.
And this is okay in this case because there's an inverse to the conjugate operation.
In fact, conjugate is a self inverse.
Right? a plus BI to a minus BI to a plus BI.
So because it's a self inverse, it's really easy to go through these things.
It's really easy to set up, you know, the bidirectional lens if you are familiar with the functional programming literature, the bidirectional lens that says, you know, when you make an update to some view how to propagate the update back to the original thing.
Inverses just make this easy.
And then we've got something that, like, is a view and, you know, share storage.
It has aliasing semantics, which is one of the reasons why conjugate views are backwards compatibility breaking.
So they're kind of an experiment.
Right? Like, maybe people are actually mutating their tensors after conjugating them and expecting the conjugates to stay the same.
I don't know.
So that's one of the things we need to work out.
by putting this in master.
But, like, you know, if this all works out, you know, we have an actually interesting new tool that we can use in other situations that, you know, allow us to do fusion without having to worry about the, you know, concurrency problems that lazy evaluation give us.
So, conjugate views.
They're not in master yet, I think, but Anjali Chordia has been working hard on actually landing it.
She's done most of the work on actually, you know, pushing this to the finish line.
And, yeah, I hope it is a cool feature and one that will pay off for us in the future.
That's all I have to say today, talk to you next time.
.
