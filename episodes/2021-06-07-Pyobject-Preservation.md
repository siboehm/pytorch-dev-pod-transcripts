---
layout: post
title: "Pyobject Preservation"
date: 2021-06-07
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Pyobject Preservation

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I wanna make good on a promise that I made on the very first episode of the podcast.
Namely, How the heck when we bind c plus plus eight intensors to python, can we make it so that the python object doesn't go away when the pie thermologic goes dead.
Namely, how do we preserve the pie object? This podcast is gonna be a little technical the way that I'm gonna do it is I'm gonna first explain how the trick works, which will sound really simple, stupidly simple.
In fact, but with a lot of complexity underneath.
And then we'll just go in a wild romp on various aspects of how c Python works and how c extensions work with Python to explain all of the more subtle moving parts of how biometric preservation works.
Because, yes, it does sound very simple and it is simple.
There's just a lot of teas to cross and eyes to dot.
Alright.
So where should we start? So just to remind ourselves about what the problem is, Imagine you have two objects, two ref counted objects, they're ref counted separately, you know, so object a has got some ref count.
three.
Object B has got some Revcon two.
And what you'd like to do is you'd like to set them up so that object B stays live as long as a has non zero ref count and object a stays live as long as b has non zero ref count.
So let's imagine that these are one of the objects is our c plus plus tensor with a c plus plus reference count.
And another object is our pie object representing the c plus plus tensor and it has a python ref count.
This puzzle basically evolves into how do we make sure that we actually keep the c plus plus tensor and the pie object around in the same linking their lifetimes together.
Before we explain the solution, it's helpful to think about two solutions that don't work.
One solution that doesn't work is to have the c plus plus object have a strong reference to the python object and the python object to have a strong reference to the c plus plus object.
Why doesn't this work? Well, you have a reference cycle.
You call in your class about reference counting that reference counts are very nice, but they have a problem which is that if you you have objects that refer to each other, those objects will never get garbage collected unless you break the cycle in some way.
So if we have c plus plus refer to python, python refer to c plus plus, That's a cycle.
That's straight out.
We will just never garbage collect the objects in that situation.
Another solution that doesn't work is to have one of the objects have a strong reference to the object, and another object have a weak reference to the other.
So for example, and the c plus plus object have a weak reference to the pie object, and the pi object have a strong reference to the Siebel Swiss object.
This is what PyTorch does today and it doesn't work because if all the references to the Python object go dead, the Python object will get deallocated because, well, the c plus plus object even if it has references to itself only has a weak reference to the Python object, so it doesn't stay alive in that situation.
Okay.
So how can we solve this problem? Well, we're gonna use a little trick, and the trick is resurrection in python graph counts.
What does resurrection refer to? So resurrection refers the fact that when you're doing Rev counting in Python.
If the Rev count for an object goes to zero, you can still resurrect the object from the dead by simply making sure that a new reference to the object gets taken out while you're deallocating the object.
When this happens, c Python will say, oh, object is still live and will abort the rest of the deallocation process.
With resurrection as our tool, we now have enough tools to actually solve the circular reference problem once and for all.
Here's how it works.
So in the beginning, we'll set things up just as we do today.
where we have a c plus plus object and a python object, and the python object has a strong reference to the c plus plus object, but not vice versa.
This goes on for a bit while we have references.
And at some point, the Python object is going to go dead.
whereas the c plus object is still live because that's the situation we're worried about.
When the python object goes dead, we don't immediately deallocate it.
And so we look at the reference count of the c plus plus object and say to ourselves, is this reference count greater than one? Because Well, if it's one, then it's solely owned by the python object in question.
But if it's greater than one, that means someone else has a reference to the c plus plus object.
and that means we shouldn't kill the Python object.
So when this happens, we will abort the the allocation and we will flip the ownership so that the c plus plus object owns the python object instead of vice versa.
Thus saving the python object from getting deallocated and, you know, because it has no incoming references, giving it the ownership in the only way that's possible.
There's one last thing.
which is that c plus plus reference counting traditionally doesn't support resurrection because it's kind of a difficult thing to do in a thread safe manner.
So what c plus plus so what we'll do is if I ever use my c plus plus object to take out a new owning reference to the Python object, And this shouldn't be too hard to do because you had to call some API with a c plus plus object to get the Python object in question.
then you can actually just flip the ownership back so that the python object refers back to the c plus plus object once again.
And then you can do this as many times as you want, as many times as the Python object goes dead while a c plus plus object is still live.
And so we wrote this up in a patch We put it in Pytorch master.
And so now if in Pytorch master, you say assign a variable to the grad field of a tensor, The grad field, by the way, is stored in c plus plus.
So it isn't a good old fashioned pie object field.
It's a actual field in c plus plus.
So your storage sensor in there and then you delete all references to it from Python, you will still retain, for example, the dict properties that you put on the TensorIN question.
So no more lost Py objects.
So that's it.
That's how biologic desert preservation works.
Feel like you want a little more.
Perhaps, Well, let's dig into a little bit about why this actually works.
And the first question that you might ask is, hey Edward, So it's kinda cool that there's this pie object rev resurrection mechanism.
By the way, it was Sam Gross who came up with his technique.
He was the one who told me about it.
and let me actually implement this in this way.
So why does resurrection exist in pi Python in the first place? And the answer is finalizers.
What is a finalizer? So in Python, you have all these objects and sometimes they go dead and sometimes you wanna clean up after an object after it goes dead.
For example, if you open a file when the file object goes dead, you might wanna close the file in that situation.
Of course, what you really should you do in that situation use a context manager to guarantee the file gets closed.
But if you don't use a context manager, the file will still get closed when it you know, gets deallocated because of the finalizer.
So Python supports arbitrary finalizers.
You can write whatever code you want.
If you want to write a Python object and write some finalization code on it, you can just write the magic method underscore underscore Dell on it.
Cool.
So there's a problem.
Right? So finalization is when the object is dead and we're trying to get rid of it.
So the finalization, who can do anything? So what happens if you accidentally, like, you know, or or purposefully, you know, put out a new reference to the object you're being finalized somewhere else.
Mhmm.
Well, that's a bit that so for a while, this was kind of skippy.
And eventually, there was this PEP safe object finalization, which said, okay.
What we will do is we will resurrect the object when this happens, so we will make this a valid thing to do and we'll just mark the object as, oh, this object has been finalized and so I'm never gonna finalize it again.
So so you have the environment that an object only gets finalized every once.
So this by this way, like, you know, we don't have to worry about audio being in strange, half deconstructed states, and then escaping into the outside world.
Because we we just run the finalizer, finalize the resurrects it, we just stop deallocating and then we wait until later when the object actually becomes dead to deallocate it.
So this is why resurrection works, but it also poses a question for pi object preservation, which is if finalizers can only run once, I better not run my finalizers when I'm doing this one of these resurrection things.
And actually, it's a little difficult to arrange for this to be the case.
because let's explain how deallocation works in c Python.
So in c Python, when you define a any type of Python object, There are a bunch of TP fields which define the various behavior you wanna do.
So there's like TP and Net that says what to do during construction.
And for our purposes, there's one that's very important TPDalloc.
What is TPDalloc? It just says how to deallocate an object when you call into it.
And so when you, like, write a c extension a custom pie object, you'll typically provide a t p d l look that, you know, like, looks into the c plus plus fields or whatever it is.
you're implementing in the Py object and actually deletes them so that, you know, we deallocate them.
And at the other day, it actually also deletes the Python object altogether.
Okay.
So that's kinda cool.
What about when you sub class a Python class in, you know, say Python? And this is relevant to tensor because we don't actually let people use the c bound object called tensor based directly.
We actually sub classified a tensor.
Well, Python subclasses have their own special deallocation implementation called sub class deallic And this deallocation method sort of takes care of all of the random things that, you know, Python objects actually support.
So there's a good reason why we sub class tensor into a Python sub class, which is that if we didn't do that, many things that people would expect to work on objects such as, you know, writing to arbitrary fields on the object, using weak references, doing finalizers, all those things wouldn't work.
Right? Because those things are actually handled by the implementation of the Python sub class.
And we would have to, like, manually replicate them in our c implementation if we wanted them to work without sub classing.
we got a problem.
Right? So what happens when I deallocate an object, I call the t p d Alloc for the most specific sub class that the object is in question.
And that's gonna be the Python sub class in the case of Tensor.
And what does it do first? Well, it runs finalizers.
and I don't want to run finalizers because they might be resurrecting this object.
So what's a poor person to do? Well, we need to somehow override the t p d l up for all sub classes of tensor bass to make sure that they first check if resurrection is gonna happen and bail out entirely before the deallocation process has chance to mark the object as having been finalized.
Do you have a way to do that? Fortunately, yes.
In Python, you can define a meta class.
what is a meta class? A meta class is a way of customizing the behavior of classes when they get sub class.
So if you imagine like a class constructor is something that gets called when you construct an object.
A meta class constructor is something that gets called when you construct a class.
as part of the meta class hierarchy.
So do we do? We define a new meta class for tensor base.
And so when we sub class tensor from tensor base, the methanol class gets run, and what it does is it just overrides the TPD ALOC to replace sub subclassdialOC with our own THP variable subclastialic.
It actually looks very similar to subclastialic.
Right? It still needs to clear out slots.
It still needs to deallocate the dictionary.
It still needs to run finalizers.
But before all that, it checks if we are going to resurrect the object by looking at the rough count of the c plus plus object.
It's a little unsatisfactory because I actually went ahead and looked at c Python and copy pasted all a code for sub class d ought to make this all work out.
But it's it it works out in the end because actually a lot of python binding code like python for example replicates this.
Because remember what I said, if you just do a very simple c object from Python, you don't get dictionaries, you don't get slots, you don't get any of that stuff.
So you want that all working, you have to actually write code for it.
And so, Citon, for example, does replicate all this logic so that it looks like it without you having to sub class from Python.
So that's one of the complications that arise from doing a sub class preservation.
What's another complication? So another complication is that weak references are a little bit of a problem.
So I said earlier that we need to be able to intercept whenever a strong reference is taken out to the pi object from the c plus plus object because we need to fix up the ownership in that situation.
If the c plus plus object owns the pie object, I need to flip it back around so the pie object owns the c plus plus object.
And ordinarily, it's easy to enter pose on this, but there's one case you can't enter pose on it, and that's a weak reference.
A weak reference lets you take a reference to an object that, you know, will go dead if that object goes dead.
But if the object is still alive, I can use it to manufacture a strong reference into the object.
and there's no way to hook into this behavior.
So if someone's got a weak reference, they can get out a reference to the pie object even if I'm still in this flip state.
where the c plus plus object owns the pie object.
This is mostly harmless and less than the c plus plus object goes dead while the strong reference from the weak ref stays live, and then you're in this awkward situation where the c plus plus object gets deallocated because there's no resurrection for c plus plus objects.
Fortunately, there's a simple workaround for this situation.
You just need to, like, ask to fix the reference direction.
And so I added a new method to Tensor that lets you do that if you're using weak references.
But actually, none of our tests failed because of this, so I'm suspecting that no one's actually gonna run into this in practice.
One last thing.
So so Python has this thing called a garbage collector, and actually what it does is it makes it so that if you do have cycles, and entirely python objects, you can actually garbage collect them in that situation.
So they're not actually gonna be lost to these either to forever.
By the way, this doesn't apply for c plus plus shared references.
If you have a cycle there, you're just flat out of luck.
So g c cycles are kind of interesting in Python because We also need to handle them correctly under the assumption of resurrection.
Right? If I have a cycle in Python, but it turns out that if I were to deallocate this object, then I would have resurrected it from, you know, some c plus plus object that's live, that Python object needs to be treated as a root Right? I can't actually deallocate the cycle because that would just leave everything in a broken state.
But the way that the, you know, cycle the way that garbage collection works is if I try to resurrect it at the point in time I'm deallocating, it's too late because I might have actually started deallocating all the other stuff in the cycle because Python is just gonna be breaking the cycle using TP clear.
That's the way you break cycles.
So what's a poor person to do? Well, all you need to do is make sure that when Python on is doing garbage collection.
Any object that is resurrectable gets treated as a root.
And ordinarily, AGC just has a fixed set of routes that it knows to traverse down to find where everything is, but Python is special.
It needs to do a first pass a pre pass before the actual traversal pass in GC to determine what all the routes are.
And this makes sense because, you know, you could have arbitrary references to pie objects from random places in c plus plus that Python knows nothing about.
And so in general, Python doesn't know what your routes are.
So it simply defines routes to be any object that has a ref count greater than all the ref counts coming into it from other Python objects.
So if you just make sure that something gets treated as a root and that's pretty easy to do, you just don't traverse its members in that situation, then you're all good.
And so we also, not only do we override TPD allot, but we also override TP traverse in the meta class to make sure we check for resurrection before we traverse and hit the sub members.
Okay.
So that's how Py object preservation works.
I'm hoping to release a little sample open source project that shows you how to do this trick you know, in a very compact way because I think this will apply to any project that is binding c plus plus objects to Python.
That's all I wanted to say for today.
Talk to you next time.
.
