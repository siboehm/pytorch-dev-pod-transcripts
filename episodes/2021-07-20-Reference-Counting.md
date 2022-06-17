---
layout: post
title: "Reference Counting"
date: 2021-07-20
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Reference Counting

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk about how we do reference counting in Pytorch.
might think of reference counting as something that isn't all that interesting, especially in c plus plus where there are plenty of classes like shared pointer that allow you to do reference counting without having to think very hard about it.
Well, there's actually a lot of subtleties doing reference counting in Pytortch, and they wanna talk about a few of the things that are going on here.
So one of the very first things that you figure out when you're looking to reference Conning and PyTorch is that we don't actually use shared pointer for most things.
Instead, we use this thing called intrusive pointer.
intrusive pointer is the term of art for reference counting schemes, which store the rough count for an object directly on the object itself.
So this is in conscious to share pointers in c plus plus, which work on any type of object.
And the way they do that is the reference count is stored in what's called a control block, which is allocated separately from the reference counting question.
Of course, if you use to make shared, on the control block and the actual object and question will be done together in one allocation.
But in general, you when you have a shared pointer, it's actually two pointers.
one pointer to the control block and one pointer to the actual object in question.
So that's a little wasteful and it also makes it difficult to take a raw pointer and convert it into an owning pointer.
So in PyTorch, we implement all of our reference counting using intrusive pointer.
So the intrusive pointer Stores the rev count on the object, you have to inherit from a intuitive point pointer base, which says, hey, here's where the rev count is.
Here's the memory layout that intuitive pointer accepts.
expects, and then Intrusive Pointer is just, in fact, the actual, you know, Smart Pointer class that handles the reference count increment and decrement when things go in and out of scope.
So the tensor type that you all know and love is exactly simply a wrapper on top of an intrusive pointer to the tensor impulse which actually comes the contains the Tensor data in question.
And Tensor Imple has a very minimal API and then Tensor the wrapper class actually has a ton of extra methods to find on it, which, you know, lets you do all of the good old fashioned method calls that you wanna do in PyTorch.
So the rough count on intrusive pointers is atomic.
So it means that Pytruge does work correctly in a multi threaded set setting, but it also means like its shared Pointer breathing, atomic operations are actually quite expensive, and that also means that intrusive Pointer bumps are also expensive.
Why, by the way, are atomic rough count bumps expensive? Well, the reason is that when you do an atomic operation, your processor has to actually bounce the, you know, cash line, which, you know, previously could just directly operate it on back into main memory to make sure things gets consistently seen by the other cores in question.
And that communication is quite expensive.
In contrast, Python does a lot of rough counting and people don't generally think of increasing or decreasing Rev counts in Python as very expensive.
And that's because Python Rev counts are actually non atomic, and they're protected by the global interpreter lock.
So, you know, the interpreter only runs in a single threaded fashion.
And, you know, increments and decrements that are not atomic that are not locked, those are very cheap to do.
So because tensor rev count bumps are very expensive, we actually go through quite a lot of trouble to avoid actually doing ref count bumps when we can.
And in fact, in PyTorch, when we write functions, like we write operators, Typically, the lifetime of tensor is very, very regular.
Right? In particular, is that, you know, when we call a function with much tensors, those tenses are gonna stay live for the entirety of the function because, you know, what are these functions doing? They're not storing things in data structures.
They're not destroying anything.
Right? They're just reading in the tenses as input, and then doing things with those.
So in fact, everywhere in Pyturgical, where, you know, you don't actually wanna steal a tensor in question, We just pass around constant sensor ampersand, which is just a very convenient way of writing, hey, pass in the sensor and don't actually, you know, do a reference count bump when you pass it in in this way.
Now, if you're a veteran c plus programmer, you might be thinking to yourself, hey, why are you doing a const reference to a shared pointer type which actually points to the object in question.
Isn't that a double in direction? Shouldn't you just be passing a, you know, tensor, impulse star or some sort of direct pointer to the object in question in this situation.
And really the answer is yes, you would be right In an ideal world, this is what we would do.
But remember that Tensor is a type that has a lot of methods on it, and Tensor Imple is a very bare bones type.
So, you know, when we were originally writing out the A10 library, we had this problem, which is that, well, you know, these tensers that people wanna take in a non owning fashion, while these people still want all of the methods all of the, you know, useful convenient stuff that's only on tensor and not on tensor impulse to be available in this situation.
And if you pass a tensor impulse star, well, you're not gonna get any of that information.
So, you know, at the very beginning, we were like, okay.
Well, we're just gonna cause tensor empressand.
and, you know, that'll be very easy and convenient to do, and you'll get all the API that you had before, and then the rest of this history.
So, like, everywhere you look in PyTorch, you're gonna see constant denser air descend all around everywhere.
There's also a little bit of nuance here which is that if you have a constant or ampersand, you might be thinking to yourself, hey, you know, maybe I should just pass it by value And that, you know, also whenever I get to move into the tendering question, doesn't that, you know, save me a reference count bump in that situation anyway? And certainly, if you are dealing with a function that wants to take ownership of the tensive in question, this is certainly a good thing.
But Once again, most of the functions in Pytorch are borrowing from the tensor they don't actually take on ownership.
And there's this funny business with the Itanium ABI which says that if you have a non trivial class, an intrusive pointer is a non trivial class because it has a destructor that's responsible for decrementing the rough count.
when exerts.
If you have a non fluid class, you must put it on the stack so that I can take a pointer address to it.
So I'm not allowed to pass in and then choose a pointer to a tensor and pull directly inside of a registered.
It always has to be on stack.
It's a kind of crappy thing about the ABI.
It actually is one of the reasons why unique pointer is not a zero cost abstraction.
You pay for using unique pointers instead of raw pointers that you just manually Alok and Dialok.
But, you know, basically, whenever you say constants or empressand, that's basically what, you know, people were doing anyway when they were forced to put their intrusive pointers on the stack.
So it's no worse, really.
So taking stock where we are right now.
So we've got tensor.
Tensor is a reference counted type.
It internally is represented as an intrusive pointer to a tense or simple, which actually contains the actual data for the tensor in question.
Reference count bumps in pitrotrotor atomic and therefore expensive.
And in order to get around that, most people pass them on tensors as constant or ampersand.
By the way, this constant on the constant or ampersand means that you're not allowed to mutate the reference itself.
Right? So like if I had a tensor x and then I pass it into a constant sensor and percent, you wouldn't be allowed to, you know, set x equal to y, and that would change what the binding was at the top level.
What it does not mean and what something that is very easy to get confused about is it does not mean that the tau tensive itself is constant and we're not allowed to mutate it.
You're allowed to mutate whatever you want.
Cons correctness, untensored is not actually a thing.
And this is because When we say constant tensor ampersand, we mean a constant reference to a mutable tensor, not a reference to a constant tensor, which and, you know, shared parlance would have been shared parter open angle bracket, consts, tensor, closed angle bracket.
That's just sort of not representable if you just say tensor because tensor is already, you know, an intrusive pointer to a tensor impulse.
So you'd have to, like, come up with a different type.
like constant tensor in that situation, which, you know, might not be a bad idea, and there's an issue about this, and someone should go about and implement this at some point in time.
A funny problem happens occasionally when you're working with this tensor type, which is that sometimes you have a tensor impulse star Remember, one of the perks of doing intrusive pointers is you can pass around a bunch of raw pointers to the objects in question, and then you can always easily convert these into real honest and goodness share pointers.
You can't easily do that with a share pointer because while, you know, you need to somehow get out the control block.
That's why enable shared from this is a thing that you know is an extra bit of information that records where the control block is so you can always get to it when you need it.
So your problem is you've got one of these raw tensor impulse and you want to pass it to one of these constensor ampersense that I said is all over the code base.
in Pike Church.
And here's the problem.
To do this, you actually need an honest to goodness Tensor Class.
Although the Tensor Class is, you know, representationally equivalent to a raw pointer because at the end of the day, contains a c ten inches of pointer.
And what is a c ten inches of pointer? Well, it's just a raw pointer with a bunch of specialty structures.
c plus plus does not allow you to actually interchangeably, you know, convert between these two representations.
So like you're kinda stuck, right, to actually pass a tensor impulse star to a constant tensor ampersand, you have to somehow manufacture a tensor.
But manufacturing a tensor you know, ordinarily gives you a rough counted owning object that is obligated to destroy the tensor you know, decrement the ref count when the tensor goes out of scope.
So it seems kinda like you're out of luck.
Right? Like you want to create a non owning constancy reference, but you can't do it because well, you know, you have to make a tensor and tensor's getting in the way.
So Scott Wachok had a really good observation about how to solve this problem.
Right? So remember that the problem is that if we create a tensor, well, one is that, you know, ordinarily you have to increment the rev count when you create a tensor, but you could imagine skipping that.
But then when you destruct the tensor, the tensor will actually decrement the rev count.
Right? So you've got two rev counts you need to somehow get rid of.
But Intruded Pointer actually has a condition and it's deallocation.
And the condition says that we only decrement the rev count if the intrusive pointer actually is non null.
If the intrusive pointer is null, we skip the decrement altogether.
And this behavior in the destructor gives us an out.
Right? What it says is that if I manually clear the intrusive pointer, before the destructor of tensor runs, then the destructor of tensor will see that the pointer is null and it'll skip the deck graph.
So all I need to do is be able to release and introduce a pointer without recommending the ref count and knowing out the value on the inside.
and I can get by Scott free.
And this is the idea behind TensorRev.
So how does TensorRev work? So TensorRev is a class It contains a tensor as its member, but it's intended to be a non owning version of tensor.
So you are able to construct these without incrementing rev count bumps And when you destruct these, no f comp bumps happen.
On construction, what you do is you take a tensor and you take the raw pointer for that tensor and you manufacture a new tensor object without actually incrementing the rev count.
Intrusive pointer actually has an API for doing this.
It's like don't increase rev count tag, and the constructor.
It used to be private, but, you know, we made it a little less private so that we could do this particular thing for tensor reps.
And then when we destruct the object, well, destructors for child classes run before parent classes.
So in the child class destructor for tensor ref, what we do is we release the pointer.
So what release does is it sets this interest corner to null and skips the rev count bump.
And now the parent destructor, which, you know, is going to process the members in the class in question, namely the tensor, we'll see that while it's a null pointer, so there's nothing to do.
So you bypass the increment rev count and decrement rev count in both cases.
And once again, what was the point of doing all of this? Well, now I have a way of giving a tensor impulse star I can create a tensor constant or ampersand.
Right? I do that by creating one of these tensor graphs, which internally contains a constant tensor ampersand.
And that's the way that I can actually then call these functions without having to do any reference gun pumps.
So this is a pretty good eye cool idea, and we actually never implemented it.
And the reason we never implemented it was because, well, you know, TensorF is an entirely new class.
C plus plus doesn't have dot overloading.
That is say, there's no way to say, hey, given a class, here's what the meaning of all dot full operations means, because then I could just forward it to Tensor.
So actually, we'd have to cogenerate all of the same methods that used to live on tensive on tensive ref as a wall.
That was kind of a pain and so no one has gone around in doing it.
However, Megan Lelee has been working on a similar concept, optional TensorF.
So what is optional TensorF? Well, optional TensorF is for those situations where you want to optionally pass a night sensor to one of the colonels in PyTorch or maybe there's no denser at all.
Previously, we implemented these as a stood optional tensor, but there's a problem with this implementation.
Do you see it? Stid optional tensor with no extra references or pointers or anything like that implies that you're getting an owning reference to tensor.
So in fact, to call a function like this, you have to do a reference count bump.
That's bad.
And, you know, we kind of mess this up and we're trying to fix it with structured kernels.
So, functional tensive ref doesn't have this problem.
It also is a little more efficient than optional tensive because optional tester, the optional class is obligated to store whether or not the tester is full or not by a separate bullion.
but, you know, we can actually just represent that as a null pointer, tensor inside optional tensor ref.
And finally, optional tensor ref doesn't have the problem of the API.
because, well, you expect to have to use arrow notation whenever you're accessing an optional object because you don't know if it's null or not.
So there's a lot of stuff that goes into reference counting in PyTorch.
And if there's one thing that I want you to take away from this podcast, It's that atomic ref counts are expensive.
So avoid them whenever you can.
That's everything I wanted to say for today.
Talk to you next time.
.
