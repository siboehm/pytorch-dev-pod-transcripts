---
layout: post
title: "Weak References"
date: 2022-06-06
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Weak References

Hello, everyone, and welcome to the Pyturbine podcast.
Today, I want to talk about weak references.
Some useful background for today's podcast, So we have a podcast about reference counting from way, way back then.
Still relevant.
If you haven't listened to it, give it a listen.
I'm not gonna go over reference counting basics.
and you might also be interested in the Python resurrection podcast.
Also, just check the links in the podcast.
That one's not that one's optional.
You don't have to listen to that one, but it's some useful context as well for discussion about weak preferences.
Okay.
So weak references.
What are they? And what are they good for? So a weak reference is a reference to an object that doesn't keep the object live.
So let's imagine that you've got a tensor and it's got a lot of data in it.
And, you know, you want to be able to store a reference to it because you're keeping it in a cache or something like that.
but you don't actually wanna hold on to it because maybe you're just cashing it.
Right? So if everything is done using it, then your cache is, you know, never going to actually let the sensor get freed in that situation.
But the cache is purely advisory.
If no one's using the object anymore, then you would like the cache to automatically free it in that situation.
You don't want the cache's reference to the object to be strong.
You'd like it to be weak.
Another common situation that this sort of thing shows up in is let's say that you have a cache that is keyed by a tensor.
So you're mapping a tensor to another tensor.
So let's think about the key tensor in this situation.
This key is basically in the hash map so that we know how to correspond.
the, you know, input tensor to whatever the cache output is.
But once again, we don't want to keep this input live.
If all the references to the input are dead, then there's no way I can ever actually pull out that tensor from my cache.
So I really don't want the cache to keep it live in that situation.
One last example for weak references.
So in in Python, object manipulation is very flexible.
So you've got all these objects lying around and you can basically mutate them willy nilly however you like.
Right? You can like add extra fields, do whatever you want, unless the object, you know, doesn't support underscore Dict, Python supports adding arbitrary attributes to objects.
But there's a problem to this.
Right? The attributes on the object form a sort of global name space.
So if you you're, you know, using, you know, one name like say name, for example, for your own nefarious private purposes.
And someone else wants to also put something else on the same field name.
Well, that's gonna be a conflict and your code is just not going to work.
And so because of the situation, it's not really safe to just arbitrarily write random attributes onto tensers.
You'd kind of like them to be, you know, some private and some way.
Now, of course, you can mangle the name of the attribute to make them private, but there's another way you can also implement this.
And that is once again, using a weak map.
you just have a week map mapping, you know, any given tensor to the attribute you wanna store for them.
And as we said earlier, we do want the entries in this map to get garbage collected if the sensor goes dead.
And, you know, that's what a weak map.
exactly what do.
And similarly, because we have separate maps for all of our various users that wanna store metadata, then you actually, you know, don't ever have a possibility of conflict because each map is its own heap allocated object and, you know, they're not being addressed by some string name.
something else that's really good about doing it this way is that you can also just delete the entire week map when you're done, and then all of those attributes go away.
So you don't have to, like, worry about, well, you know, I'm done with all of my private attributes.
How do I get rid of them at some later point in time? You know, you just use a week map to do that.
Okay.
So weak references, hey, they're kind of useful.
So we do support them in Pytorch in several ways.
One is in c plus plus.
Obviously, if you use the Share Pointer, Smart Pointer type, Share Pointer comes with built in support for weak references.
Also, our intrusive pointer, see our previous podcast.
That also supports weak references.
And of course, Python with Python objects, they also support weak references.
So there's actually two weak reference mechanisms either a c plus plus mechanism or the Python side mechanism.
And you can use either one if you have an object that's bound in both places.
So what I wanna do is I wanna explain a little bit of how how these are implemented and then some consequences of these implementation decisions.
Let's get down to it.
So how are c plus plus week references implemented? Well, when we talked about reference counting, we said reference counts were a field on an object saying how many references there were to the object so that when the, you know, field was still, you know, positive, that meant the object was live.
And when that count goes down to zero, now we know the object is dead.
So weak references are just a, you know, extension to this, where not only do we keep a strong reference count, We also keep a weak reference count on the object.
So the weak reference count as its name suggests counts how many weak references there are into the object.
Now that do note that when I have a weak reference account, it's actually not only weak references.
There's actually one extra weak reference and that's for the strong reference count on the object.
So the environment here is as long as the strong reference count is greater than zero, then my weak reference count has one is at least one where that one is from the strong reference count.
And then you can have as many extra weird references to the object as you like.
So how do these two reference count fields interplay with each other? Well, the algorithm looks like this.
So long as the strong reference count of the object is greater than zero.
The object is live.
And when the strong reference count goes to zero and, you know, this zero this testing if the star reference account has gone to zero is an atomic instruction.
When it goes to zero, that is when the object becomes dead.
So no matter how many weak references you have to an object, it doesn't matter.
Right? Week references don't keep an object live.
Only strong references keep it object live.
So when all the weak reference so when all the strong references are gone, then we kill the object and we say, okay, we are done with this object.
However, ordinarily, when we wanna deallocate an object, we would just go ahead and free the memory associated with this object.
that's not okay.
We've got a bunch of weak references to the object that are pointing to this memory.
Now I just go ahead and free that memory.
There's no way for those weak references to know, hey, you know, there's no object here anymore.
I can actually give you a strong reference.
By the way, when you have a weak reference to a still live object and you say, hey, I would like a strong reference from this week reference.
You I'd like to dereference the week reference all we do is we attempt to automatically exchange the strong reference count with one greater than the strong reference count.
and that will succeed so long as the strong reference count was in zero.
And if it was zero, then we'll just say, hey, there's no element available in this situation.
So we've got these weak reference counts, but they need to be able to access the, you know, reference count fields that are stored on the object.
Remember, this is an intrusive reference count in our case or the control block in the case of a Share Pointer.
And so if I just go ahead and deallocate that, then that's no good.
Right? I don't actually have the data anymore.
It would just be in ASAN violation in that situation.
So what I do is I actually keep the object live.
Now, wait, you might be saying, that sounds very silly.
If I keep the object live, then what's the point of having the weak reference distinction? Aren't I supposed to deallocate the object in this case? And indeed, for, you know, objects that are sort of stored all the areas stored in line, weak reference are kind of useless in this situation.
And so with shared pointers, the way this is dealt with is actually the reference counts are stored in an extra object called the control block.
And the control block is the only thing that gets stays live.
You actually deallocate the object in that case.
As long as you didn't use make shared, that is to say, which allocates the object and the control block together.
But for an object like tensor, we have something else we can do.
Right? Because the tensor object itself doesn't contain all that much data.
It is it is a kind of fat object and it's got a lot of fields on it, but really most of the data usage of a tensor is coming from the data, the actual tensor floating point data that is associated with the array in question.
So all I need to do is I just need to deallocate that.
And then I'll have a little stubby, you know, tensor data structure left, which, you know, is not which is taking up some space, but it's taking far less space than the actual tensor data in question.
And so we've got a method on our tensor objects that does this It's called release resources.
So just to go over the algorithm, first, we, you know, have the strong ref count is greater than zero.
We do a bunch of stuff.
when the strong reference count goes to zero, we go ahead and release resources if there are still sorry.
We go ahead and release resources.
Right? Because those are the resources that are not being used anymore.
And then as soon as the weak reference count goes to zero oh, by the way, when the strong reference count goes to zero, we also decrement a week reference count by one.
Right? Because remember there was one week reference count associated with the strong reference count.
So when the weak reference count goes to zero, then we know there really are no pointers into the data, into the object in question, and now we can actually free it.
from the heat.
Alright.
So that's cool.
So that's how c plus plus sides of portfolio references are are implemented.
You have to allocate an extra field for maintaining the weak reference count, and then there's a bunch of extra stuff that happens at the allocation time.
In the common case, when you deallocate a sensor, there aren't any weak references to it.
So the strong reference count goes to zero, that causes the weak reference count to go to zero, and then we immediately delete the object in that situation.
We are that we actually have an optimization for this courtesy of Scott Wall Chuck where we don't have to do the atomic compare and exchange anymore.
You you just do a relaxed load on the loop count and check if it's one.
And if it is, you just go ahead and delete it in that case.
Okay.
So what about Python? So Python also implements weak references, but it actually does them in a quite different manner.
And Python's implementation works because Remember, Python has a global interpreter lock, so it actually doesn't need to work in a multi threaded setting.
I talked a lot about avtomX, in the in the c plus plus side of things.
And really, c plus plus's implementation is by and large, you know, sort of it has to look this way because it's supposed to work in multi threaded setting.
So how exactly does the weak references work in Python? Well, it's pretty simple.
Every object that is able to be refer referenced as a weak reference has an extra field called the weak reference list.
What exactly is the weak reference list? Well, it's literally a list of all the weak references that point to this object.
So a weak references in Python is a special object and So whenever you create one of these to point to an object, we actually just go ahead and put that object on this list.
and, you know, that would be hella unsafe in a multi threaded environment.
But in Python, it's fine because there's a global interpreter lock, so whatever.
And so now, these week references don't actually don't actually increase the true Python Revcount.
So Python Revcount does a normal thing.
When it goes to zero, as part of the deallocation process, we go ahead and go through all the weak references point to this object and say, okay, well, you are no longer valid, so you can't you can't use this weak reference to go ahead and run this object.
And because, you know, we know what all the weak references to this object are, we can also go ahead and run finalizers.
So that's that's also when finalizers get run-in Python.
We just iterate through all the weak references.
Those weak references can have finalizers attached to them, and that's just some code we execute when we do it.
By the way, the fact that finalization can resurrect an object because, you know, finalization is just arbitrary Python code.
Maybe when you're done finalizing, the reference count has gone back to one or greater that is exactly what we're using to implement, you know, tensor, high object resurrection, which we talked about in a previous podcast.
Okay.
So that's about how python sideway references work.
So let's talk a little bit about some consequences of these implementations.
So one thing to know about is that when you use weak references in Python to specifically do tensors, you have an extra because of Python object resurrection, there's a little extra work you have to do.
So the work you have to do is there's a private method on sensor called fix wecrap.
And what it does is it makes sure that the sort of ownership partner between the Python object and the Tensor object looks the correct way.
Let me explain why this is needed.
So I mentioned that we've got this thing called Python object resurrection, which says that when a Python Tensor object would have died, we check if the c plus plus object for it is still live.
If it is, we go ahead and flip the ownership pointer so that the c plus plus object owns the Python pointer.
And whenever we take out a new python reference to the project making it live again, we go ahead and flip the reference back.
Well, the problem with weak references in Python is they constitute another way of accessing the Python object that it might be ostensibly sorry, a Python object that isn't a normal, you know, sort of give me a tensor from the Python API bindings.
And most importantly, This way of referencing the Python object is not interposable by us.
So we have no way of seeing when this sort of thing happens.
and then going ahead and flipping the ownership pointer if it's necessary.
So you have to tell us this yourself.
So this is something to be aware of.
if you're working with weak references.
And if you're working with weak references in Python, you probably want to do them with tensors.
So this is something you need to know about like.
It's very, very important to do.
Another consequence of this design is so I mentioned that release resources is about releasing resources that, you know, sort of take up a lot of space.
when the strong reference count goes to zero, but maybe there's still a weak reference count.
Release resources is a virtual method because there may be multiple tensor subclasses and they might have different resources that need to be deallocated.
So it's actually and this was discovered by Scott Walchuck.
It's actually quite a performance problem to always be to always be running the release resources method whenever a strong reference count goes to zero.
Because most of the time, there aren't any weak references.
So you can just go ahead and delete the object entirely and, like, that would be fine.
Right? That would also do the same thing.
And in particular, the delete method would not actually well, okay.
It's also virtual, but you're say, you're going from two virtual calls to one virtual call.
So Scott has a patch that basically makes the call to release resources optional.
It only gets called if we're actually in the situation where we're trying to keep the object live for weak references but we know that all the strong references are dead and we want to delete the metadata.
So, you know, there's a lot of this kind of optimization that goes into making a Smart Pointer implementation.
And so it's it's quite tricky actually.
Like, the basic algorithm is not too hard but then you wanna, like, reduce the number of tonics and, you know, get it as efficient as possible.
And that's when things get pretty complicated.
In fact, it's it's so complicated that Scott's original version of PR has a bug in it.
And the bug in it is essentially related to how we maintain the reference counts and when we have the when we're running release resources.
Because release resources is actually it's a pretty much an arbitrary piece of code that gets run at the end of the object.
It's it's basically like a finalizer.
And so because release resources can trigger arbitrary other disruptors to run, one of the things that it can do is it can actually cause a weak reference to the tensor you're currently deallocating to be dead.
So you need to make sure that while you're running release for resources, you don't accidentally deallocate the object you're working on while you're doing it.
Right? Because you're ostensibly running release resources because it's being kept live by a week reference.
But if that week reference dies, while you're releasing the resources, you need to keep the object live until you're done running release resources, and then you can delete it.
So, you know, just the kind of thing to be worried about.
Okay.
That's everything I wanted to talk about today.
See you again next time.
.
