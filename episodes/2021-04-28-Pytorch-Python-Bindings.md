---
layout: post
title: "Pytorch Python Bindings"
date: 2021-04-28
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Pytorch Python Bindings

Hello.
My name is Edward, and this is episode one of my podcast about Pytorch Things.
I'm not really sure how this is going to work out or where I'm going to go with this, but for now, the idea behind this podcast is just to, you know, be a casual form for me to talk about, you know, various aspects of the Pirate project.
No particular organization.
Today, I wanna talk a little bit about how we bind Python to PyTorch, that is to say, you know, the whole point of Pytorch is to provide an object called a tensor that people can use.
And, you know, to make this tensor object available, From python, we have to do bindings for it.
And these bindings are actually quite intricate in some sense.
And I wanna just explain why it's not as easy as it seems and talk a little bit about, like, how we actually solve this in the project and some of the work that I've been working on recently.
So what are python bindings? Well, let's imagine that you're trying to design any sort of, you know, high performance computing library that has bindings available from a dynamically scripted language.
So if you were just writing a data structure in the language itself, you would probably just define a class for the object in question in the language itself.
and that would give you something very reasonable.
Now the problem is, you know, when you're writing in interpreted languages like Python, all of the objects need to have a very regular layout.
And it means that, you know, when you want to do something that actually needs to be very efficient, that needs to have some sort of packed layout.
Typically, the language itself won't give you enough facilities to actually define the exact data layout you need.
It's gonna be something that, you know, you have to go to a lower level language like c or c plus plus to do.
So the typical situation for anyone who's writing a language sorry, a library in this situation is you'll have some sort of data structure.
In our case, let's call this data structure a tensor And in and then you want to somehow make it possible for people to access this data structure from Python.
So you've got two objects in hand.
Right? You've got this concept of an object in c plus plus land or in c land a struct that knows nothing about Python per se because maybe you also wanted this library to be usable by other people who don't have Python.
And then you also need to somehow give a representation, a Python representation that regular python programs can understand.
And sort of this split, this split where you want it to work both in a Python agnostic context and a Python context is where some of the complexity of binding objects in this way comes from.
Now, wait, Edward, you might be thinking, hey, you know, I can bind objects to Python.
There's this cool library called Pi Mine eleven.
And all I need to do is just take my object, you know, and wrap it up in this magic class underscore template.
And then pi nine eleven goes through all the work somehow of, you know, making it possible to actually, you know, turn this object into a Python object.
And I don't know what it really does.
But, you know, something happens.
And so I wanna talk a little bit about what happens in this case.
And actually, when we talk about a type like tensor, we don't actually use pi behind eleven.
to bind it because pi one eleven does something very interesting, uses a hash map, and we don't wanna pay the cost for that.
So let's talk about what it means to make a type actually available in Python.
So we've got some c plus plus type, we've got some c struct, and we wanna make it available to Python.
So when we're writing some Python bindings, we need to define a Python layout data structure that represents the Python object in question So remember Python is an interpreted language.
All of the objects have a very regular form.
Python is rough counted.
So one of the things that every Python object needs to have is a header saying what kind of object it is and what its reference count is.
So if you, like, go and look up your c Python, you know, API notes about how to define a new define a new object It'll tell you, hey, you know, first to find this header, then you can put in your fields, and then there's description of the data type you have to do actually say what the object in question is.
Okay.
That's cool.
So you can like copy, paste some code and get this working.
And then you have a problem, which is that you've got this python object and it's not the same thing as your c struct.
So what do you do Well, you could do something like, okay, a Python object is simply a a object that contains the c plus plus object in question.
But this usually isn't really quite what you want.
because let's say that you have a preexisting c plus plus object and you wanna pass it to Python.
Right? Like, say, I allocated a tensor from c plus plus and I wanna return it from my program and actually have, you know, someone in Python make use of it.
if you just put the tensor in the python object struct directly, well, you need to somehow you know, move the data over into this new struct layout that's got this header that, you know, Python expects your stuff to have.
and you probably don't wanna actually move all of the data in question.
So, you know, the obvious thing to do in this situation is do an in direction.
Right? So instead of having the entire, you know, contents of the object stored, you'll just have a pointer.
Right? Maybe a shared pointer.
to the representation in question.
Okay.
So that, you know, lets you construct a Python object, but Something very strange will happen if you actually try to run the code in this case.
What will happen is you pass your object to Python.
You construct one of these Python objects.
You wrap it up.
You set the pointer to point to the c plus plus object in question and you got this Python object.
then the next time you decide you wanna return this Python object, well, oh, okay.
I need to go wrap up my a pointer into one of these Python objects.
You've returned that.
Notice something has happened.
I've actually returned a new object in this situation so that, you know, even though both of these Python objects point to the same underlying c plus plus object There are two different Python objects.
And if I do something like, you know, a is b, you know, the the test for object identity, in Python, Python will just happily tell me no.
They're not the same thing.
Even though the c plus plus type is actually the same thing.
So usually when we bind objects that have this notion of, you know, object identity, you know, usually objects you can mutate like ten for example, we want to also preserve this notion of object identity when we bind them to Python.
And so pipeline eleven lets you bind arbitrary objects to Python, and it also preserves object identity.
And the way it does this is it maintains a giant hash map of all the c plus plus objects you've sent through it.
So that the next time you send the same c plus plus pointer through it, it can look it up in the hash table and say, oh, this is the Python object that I used last time, let me just return that again.
And this is how everything bound with pipeline eleven is going to work.
Okay.
Is this setting off performance alarm bells for you? Because it is for me.
And this is kinda not actually you know, this is not that fast.
And if you really care about making things fast, you don't actually want to bind your objects this way.
You want something cheaper to actually implement on this.
You want, for example, to just be able to dereference a field on your object to get the python object in question.
And so this is what we did for tensor.
So for tensors, We don't maintain a hash map mapping an a given tensor to its python object.
Instead, we have a field on the tensor object and this field simply points to the Python object in question that we wanna return.
So if I wanna pass a tensor from c plus plus to Python, I just read out this field.
If it's not null, then I there's a Python object and I'll just return that directly.
If it is null, That means it's the first time I'm actually signing this tensor to Python, so I can just go ahead and allocate one of these Python objects as I would have done before.
And then I actually, you know, get this object in Python in this situation.
So that, you know, works okay.
And remember that even though, you know, allocating a new object and then setting of the tensor seems very threat unsafe, all of our Python interactions are protected by the global interpreter lock.
So actually, you know, Python takes care of all the synchronization for us.
So this works decently well and it's what we do.
One thing that you have to be careful about is this pointer that the tensor object has to the python object is non owning.
Because remember, the python object needs to keep the sensor c plus plus sensor live.
Right? So it has a strong reference from Python to c plus plus.
If the c plus plus object also had a strong reference to the python object, you'd have a reference loop.
And that's bad.
Because when you have a reference cycle in in a rough kind of language, the result will never actually ever get deallocated.
So strong reference from Python to c plus plus because, you know, if you got a Python object, you better have a c plus plus tensor backing it, and c plus plus tensor to Python is a weak reference.
Those of you who are thinking ahead might realize that there is a problem.
And the problem is this.
Because the reference to the python object is weak.
If I only have strong references to the c plus plus object, and they have no more references to the Python object, then the Python object will actually be dead and it will get garbage collected by the by the c Python interpreter.
So that's not so great.
And, you know, you kind of are wondering, well, what about this stale biologic pointer in this case? Well, fortunately, we can actually define what the destructor for python tensor object should be.
So we just say, oh, clear out the pie object field from the tester when this happens.
But this does mean that something very strange can happen in the situation.
namely, if you have a tensor and you send it to Python and then at some point all the Python references dead are dead The next time you send it to Python, you'll get a completely distinct object.
Now granted, it's kind of difficult to notice when this has happened because while the old object isn't around because you promised that you weren't gonna have any references to it.
But know, if you, like, for example, took the idea of the object, the idea would be different between the two versions.
And more importantly, And one of the reasons why I've recently been working on a patch to change this behavior, if you actually had some Python data stored on the tensor.
For example, all all objects in Python, you know, you can add arbitrary attributes to them after the fact using the underscore underscore Dict attribute.
Well, if you went ahead and added a bunch of these things to the tensor, and then expect it once you saved it in c plus plus.
For example, if you were saving it for backwards, one of the most common cases when we'll save a tensor in c plus plus and it will outlive its python equivalent.
You won't get that information when it pops back out into python.
and we have a bug tracking this issue and people don't really like it, although it's, you know, it's kinda hard to solve a problem like this.
So next time, I wanna talk a little bit about how we are going to solve this.
And it's actually pretty nifty.
It's using a trick that Sam Gross, one of the original PyTRISH developers came up with, and I'm eager to share it with you next time.
See you.
.
