---
layout: post
title: "Tensor Subclasses And Liskov Substitution Principle"
date: 2021-09-15
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Tensor Subclasses And Liskov Substitution Principle

Hello, everyone, and welcome to the Pytruch Sturm podcast.
Today, I want to talk about Tensor Subclasses and the list of substitution principle.
If you haven't seen it already, I recently posted the state of Pytorch Core September two thousand twenty one edition, which basically talked about all of the things that were going on inside titles core right now.
And one of the things that you may have picked up reading over this is that we have actually got a lot of stuff going on related to tensor sub classes.
that is to say, you know, sub classes of tensor that add more different kinds of behavior for any sorts of things you might want to do.
there's a ton of things going on here, like linear operators, like debug tensors, like phone torch, And I wanted to pull open the cover on one of the things that we've been thinking about when designing how this ecosystem should look like, and that's the list called substitution principle, which says some things about when it is permissible to sub class some object and when it is not permissible to.
Okay.
So let's just dive straight into it.
So what is the list called substitution principle? So you may have learned this in, you know, your undergrad class about object oriented programming, and the definition you heard probably sound something like this.
If s is a subtype of t, then any t may be replaced with s without altering any desirable properties of programs that were previously using That's a bit of a mouthful.
So let's look at an example.
Let's suppose that we have some class that implements say bags.
So bags are sort of onerous collections of items.
But unlike sets, you can have multiple copies of an item in a bag.
Right? So, like, I might have three apples and two oranges.
And if I had a set, I could only say that I have an apple and an orange.
But in a bag, I can say I have three apples and two oranges.
Now, if I have an implementation of a bag, I can easily reuse this implementation to implement a set.
All I have to do is sub class it and say, well, whenever I insert things into the bag, if I already have the thing in the bag, I'm just not gonna insert it.
in that situation.
No problem.
So this sub classing works.
I can use inheritance to implement sets in this way.
and it violates the list of substitution principle.
Why does it violate the list of substitution principle? Well, imagine that you've got an algorithm and, you know, it wants to do some sort of counting of objects and so it was using a bag inside its algorithm to, like, put things in and then at the very end, read out what the counts of things should be.
If you replaced the bag with a set, which we were sort of thinking about is a set, a subtype of a bag.
If we replace the bag with a set, then when I ran this algorithm, I would only ever count up to one for any given item that I was looking for.
And that probably isn't what my algorithm wanted to do.
Barbara Liszkov gives another example, which is that In the old days, when people were sort of just figuring out this object oriented programming thing, people would make claims like queues and stacks or subtypes of each other.
Why did they say that? Well, you know, a queue and a stack have a push operation and a pop operation? And so, you know, the methods are the same.
So, well, you know, they're structurally indistinguishable from each other.
Right? Like, they just have the same methods so you can use one or the other.
And Parva was like, well, but that doesn't make any sense.
Right? Like, if I had a program and it's using a stack and then I replace the stack with a q, my program's gonna do something totally different.
My because, you know, last and first out and first and first out are totally different ways of going out doing things.
And probably my program wouldn't work at all if I replace my stack with a q.
So the moral of the story behind Liscoll's substitution principle and why, you know, like, we love to teach it in the undergrad CS curriculum is because it shows people that, hey, self classing is not the same thing as subtyping or behavioral subtyping as Lisgov like to call it in the later days.
Right? Like, just because something has the same interface doesn't mean they're actually substitutable.
You actually have to say something about what the behavior of the program is in these situations.
So I remember learning about the list of self distribution principle and thinking to myself, well, that doesn't sound too complicated.
You know, like, that this seems like a very simple thing to abide by, you know, what's the big deal.
And, well, Maybe it is.
But in fact, you know, I would say LSP has spawned a ton of debate all over the Internet about like, what exactly is meant by this.
And it's not exactly straightforward to apply the principle in any every cases.
In fact, there are some very embarrassing situations where very famous software projects have violated LSP and discovered it to their detriment later.
Malth Gommer's relates to me a very fun story from Numpy's history, which is that there's this class in Numpy called Numpy dot Matrix.
it's a sub class of ND array, so it was at least originally intended to be usable in any situation where an ND array was.
And it's basically a specialization of NDRA for the matrix situations.
Right? Two d, and what they did was they were like, okay.
Well, because these are matrices, we're gonna make multiply, like just the normal asterisk operator, meaning matrix multiply in this situation.
Well, even though Numpy dot Matrix has the same API as Numpy n d r a, it totally violates LSP because, you know, anywhere I had some Numpy program that was originally expecting to have an Endy Array, and expecting the star operator to give me point wise multiplication.
If I sub in a non pi dot matrix, I will suddenly get matrix multiplication.
And I'll probably just get errors in this situation.
And my program will not behave the same way and like it will have none of the, you know, properties that I wanted to have.
So as a result, like every, you know, like, serious numpai function.
And the ecosystem first casts everything to end year race just so that, you know, they don't have to worry about someone passing a Numpy Matrix.
You really shouldn't use Numpy Matrix if you you can get away with it.
So what I think makes LSP so controversial is that we said that you can replace any t with an s without altering desirable properties, but we didn't really say what is meant by desirable property.
Barbara at least meant what she meant by properties was that if you were only using the API defined by the supertype, you couldn't see the difference between using a t versus using an s.
And this is a very reasonable definition, especially in an academic context, but Well, in actual programming languages like Python and c plus plus, there are a lot of ways you can interact with an object.
So if you say every operation that was possible on the supertype needs to be preserved by the subtype Well, in practice, there is basically no change you're allowed to make.
Like, as a simple example, in Python, I can ask what the type of an object is.
And if I sub class my type, then I will get a different sub class in this situation and therefore it is observable that there is a difference in this situation and therefore no sub class is a true subtype in this situation.
And to take the flip side perspective, I could say, well, you know, programs are meaningless.
It doesn't matter what a program does.
All I need is for it to be type safe or for it to not raise exceptions.
And so as long as it cracks like a duck, as long as it implements all of the methods that I expected on the original object, I have no obligation to you to make the sub class actually behave in any reasonable way.
And so a lot of, you know, monkey patching and duct typing in Python sort of is based on this idea.
Right? There's no spec.
You just subclass the object, override a bunch of stuff, and pray that something reasonable happens.
So clearly, there is a solution to this problem.
And the solution of this problem is that we shouldn't use concrete implementations of objects as the definitions of our super types instead, we should use some sort of abstract specification and use that as the basis for deciding what behavior is allowable or not.
And this is definitely, in my opinion, what list club had in mind when she said, well, you know, the LSP is all about not being observably different when you talked about it in terms of the supertype.
But of course, this was in simpler times when, you know, we didn't have tons and tons of ways to break encapsulation on objects.
But of course, defining an object specification for what a tensor is supposed to be is not so easy.
Of course, it's easier than defining an abstract specification for what a widget factory is supposed to be because, you know, Tensor has its roots in mathematics.
And one could say mathematics is, you know, very much in the business of sort of abstracting away you know, differences between objects.
But at least in PyTorch, you know, we don't have anything written down.
It's all based on off of an informal understanding of how code tends to equip tensors in practice.
And that means that you really are, you know, sort of rediscovering what it means to be a tensor every time you make a tensor sub class.
Of course, there are some tensor sub classes where it's not so hard to make a determination in this way.
Right? Like for example, there are a lot of types of tensor subclasses like logging tensors, or finite tensors or NAND tensors, where it's kind of easy to see that these obey LSP.
because all they do is they do the same thing a normal tensor would have done, but then with a little extra behavior on top.
like printing out what operators were called or, you know, testing if all the elements in the tensor are finite.
And so the spec here is that, well, everything that, like, is the tensory behavior, that's part of the abstract specification, and all the other things like the logging behavior or whether or not we throw exceptions or not, that's sort of external to the tensor specification.
And most code that you write is going to, you know, be indifferent to those extra things, the extra logging or the error reporting.
It's indifferent to the error pool reporting, by the way, because in Python, you can actually throw exceptions unlike in languages like Go.
where all exceptions have to be handled manually.
If you had to handle exceptions manually, then throwing an error would not, in fact, be a easy to add piece of behavior on top.
Then there are some types of objects which mostly obey the list cobb substitution principle.
But if you poke hard enough at implementation details, maybe not.
And a good example of this are the linear operators from g PyTorch authored by Max Bell and Dot.
What are these things? Well, the basic concept is that Tensors traditionally store all of the data corresponding to them, but sometimes there's special linear algebra structure associated with the tensor And so if you store only that or you like store that, there is in fact this structure at all.
In the first place, a lot of linear algebra operations can be run faster.
So a very simple example of this is if you have a diagonal matrix, you don't need to store all the matrix, which is mostly zeros, you can just throw the diagonal in.
You wanna multiply your diagonal matrix with another matrix that's only linear.
Right? Because you just zip through the diagonal and you're done.
So these also sort of obey the list of substitution principle in a very, you know, tight way because Well, a diagonal matrix is still a matrix which is still a tensor.
So there's still this is a relationship and mathematically, you know, anything you can do with tensor you can also do on a diagonal matrix.
And even if you don't have a kernel for it, what you can do is you can just materialize diagonal matrix do a normal dense denser and then do the operation.
But there's still some stuff that doesn't work.
Right? Like, you can't get out a data pointer to the contents of a diagonal tester and then expect, you know, the first n elements to be zero.
Right? Like, you're you're gonna get if I give you a data pointer, it's gonna be this contiguous representation, and it's not really going to you know, behave the same way you would have expected with a normal stratotenser.
And this is sort of okay.
Right? Like most code written in PyTorch in Python doesn't involve poking at raw pointers.
And so for the most part, you can generally assume that code is going to behave k in the situation.
You might still have to audit your code if, you know, like, maybe your back ending to some external c kernel.
And finally, there's tensor types that don't really obey LSP at all, like nested tensors, which wanna change the type of size and tensor so that it doesn't return just a two pull of integers, but it actually returned some nested structure saying what the size of all the various dimensions in your tensor are.
And so, technically, facilities like torch function allow for this.
You can define a torch function on an object that doesn't subclass from Tensor at all.
So there isn't even any subtype relation besides the, you know, the Python duck typing relation that all objects participate in.
but it's still rough for a tensor like this because you might still want to use, like, code that was written on normal pipers tenders in this situation.
And so you're appealing to an even smaller subset of the tensive language and even more relaxed set of invariance and properties that, like, generalizes for both nest tensors and normal tensors, and it's just generally hard to figure out what this is supposed to mean.
things get even hairier if you actually, honestly, goodness, sub class from Tensor because from our c plus plus side, we have a actually we have a very strict contract about what fields in the c plus plus implementation have to be filled in you know, with actual values.
And there's very specific concrete machine types associated with them and anyone who subclasses from Tester is obligated to fill these in in a reasonable way.
And sometimes it's not so easy to do.
But because we want to be able to inline excessors, to these fields on Tensor, we have this very strict, you know, behavioral requirement that sort of makes it a little difficult to create sub classes of tensor.
That's why you have to use underscore underscore new instead of underscore underscore and net.
It's because the underlying c plus plus tensor object has to be allocated all in one go.
There are many other subtleties that I could talk about, but I do wanna relate this to discussion back to LSB for one particular aspect, which is what should be the behavior of custom utensors subclasses be when you mix two different sub classes together.
Like, say, I have a debugging tester and I add it to a diagonal tester.
Like, what exactly should happen? in the situation.
Zachary DeVito had a good comment the other day about what it means to be compositional.
What it means to be compositional is that you don't need to look at the cross product of every interaction between classes understand what things are are gonna do.
Right? So if you have to sit down and like manually write down what it means when you cross a debugging tester with a diagonal tester, you're not compositional.
Right? You're writing this monolithic thing and you've manually worked out what the interactions between these two things are supposed to be.
If we want to be compositional, this interaction has to be worked out automatically.
But how could we actually do that? Because if I am adding these two tenses together, I probably have an implementation of adding a logging tester to a normal tester.
I probably also have an implementation of adding a diagonal tensor to a normal tensor, but, you know, that doesn't give me an implementation of diagonal tester added to a logging tester.
And of course, LSP says that actually I do have a way of getting an implementation of this.
Right? So when I have a logging sensor, I also have a normal sensor.
And so I could use that tensor in place of the tensor in the implementation that takes a diagonal tensor and adds it to a normal tensor.
And similarly, when I have a diagonal tester, I also have a normal tester, and I could just use that diagonal tester as if it were a tester into the implementation of a logging sensor plus a tensor.
And so via LSP, if you actually believe in it, which it's not entirely clear that you should.
We can actually generate a implementation that works out of the box without having to, like, deal with these cases individually, but there's a problem.
Right? Which is there's two possible ways I can implement it and their behaviors are actually going to be divergent.
And so in general, this is kind of hard to resolve.
And in fact, the only way to really resolve it in a reasonable way is to do the non compositional thing and just explicitly say what the interactions of these two tenses should be.
unless you're a monktorch.
The lesson of monktorch is that if we define an ordering between these two operations, And we say, we phrase each of these tensor subclasses as a way of, you know, sort of de sugaring, a bunch of tensor operations know a bunch of lower level tensor operations that don't make reference to your tensor sub class like this diagonal tensor turns into a bunch of operations on not diagonal tensors.
If you have the ordering and you have the de sugaring, then you can decompose these and it's in a unique way and it's compositional.
So I'm not really sure what the right answer here is in general, but my hypothesis and When I look at Numpy, I see that there are plenty of NDRA subclasses, but they mostly don't interact with each other, is that People are going to write tensor sub classes.
They are generally not going to make them compositional.
And if you do want them to be compositional, while you need to fit them into a framework, like in functorch like, you know, Jack's functional transformations.
So that's pretty interesting and I hope we can develop it in more detail and share it with you when we figure it all out.
That's everything I wanted to say for today.
Talk to you next time.
.
