---
layout: post
title: "History And Constraints Of The Dispatcher"
date: 2021-04-29
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# History And Constraints Of The Dispatcher

Hi.
My name is Edward, and welcome to the Pytorch dev podcast.
Today, I wanna talk a little bit about the history and motivations behind one of the sort of more intricate pieces of Pytronch Core, the dispatcher.
Now, what exactly is the disc fracture.
Well, the dispatcher is basically the code that when you call a function, like when you call a t colon colon ad or you call a method on an utensur, it figures out where you actually want to call it.
I've done a few talks about the dispatcher and past, and I also have a blog post talking about how the dispatcher works.
And today, I wanna do something a little different.
So if if you wanna learn more about those aspects of the dispatcher, I recommend you go check out those posts.
Instead, what I wanna do is I wanna do a little historical story about how the dispatcher came to be and what the various constraints and features we needed played out over time to, you know, make it into the system that it is today.
So to talk about the dispatcher, we first need to talk about the time before the dispatcher.
So before the dispatcher existed and before eight ten existed, PyTorch was built off of this library called t h.
And t h itself wasn't written when Patrick was written and said it's it itself came from a further back library called lua torch, which was basically the torch libraries like t h and t h c bound to lua the lua programming language.
So when Adam Pashka and Co wrote the first version of Pytorch, what they did was they just took all of the old school t h and THC libraries and wrote bindings for them for Python.
And they also wrote an autograph system and, like, data parallel support but, you know, binding these torch libraries, which previously could only be called from Lua to Python was sort of the first step on the journey here.
So to understand how these binding's work back in the day, it's important to, like, understand a little bit about how t h used to be constructed.
So as, you know, is the case today, you know, a Tensor Library involves a lot of different operations.
And each of these operations needs to be implemented for every d type you wanna support.
So, like, if you talk about an operation like add, it needs to be implemented for floats and doubles and integers and, you know, thirty two bit integers and eight bit integers and so forth and so forth.
TH was written in c.
And if you've ever written any c before, you may know that c doesn't really have any facilities for actually prioritizing over different d types.
So the way that they solve this problem was they're like, okay, we're going to define a file.
We are not going to talk about a float or a double.
We're just going to talk about some abstract type.
And then we will just include this file eight times with different settings of various macros to stamp out each version of the file.
So If you talk about a function like ad, we would have a t h float tensor underscore ad and a t h double tensor underscore ad.
and so forth and so forth.
So there'd be like eight functions.
And, you know, at the Python binding level, what they do is they wrote some generated code which basically was like, hey, you know, what's the input sensor? Oh, it's a floating point sensor.
Okay.
I'm going to call t h float.
add in this case.
So it would just be the switch statement, all the various different dispatch types, and that's how things were for a pretty long time.
And about the time I joined Facebook, we were sort of trying to figure out what to do about the internals of Pytorch And one of the things that was happening was that, you know, we had just bound the torch library and everything else was written in Python and it turned out that Pytorch was kind of slow.
And Samgross did some measurements and found that, you know, the reason why Pytorch was slow was because too much of it was written in Python.
And so what we wanted to do was we wanted to port everything into c, but not actually c because writing this TH code with its, you know, macros being stamped out eight times was actually pretty horrible.
So what we actually wanted to do was write some c plus plus.
And during this time, Zachary DaVito came up with his idea, oh, all we want is a simple tensor library that gives us a tensor type in c plus plus just like the tensor type you would have in Python with all the stuff you want, and then it'll be easy to put all this stuff from Python t c plus plus because we'll just use this sensor type and write the stuff we want in this case.
And so, Zach, sort of, it's really funny.
Like, the way Aeten got written was I think, Zach, locked himself in a room for two weeks.
And at the end of two weeks, a ten was created, and Zac went through a bunch of different designs.
He actually I remember we were chatting about this, and he was like, you know, I've gotten to this point, and I don't know if I should implement multiple dispatch or not, and we, like, talked about some of the pros and cons.
And in the end, he didn't decide to do that.
And so what Zac did was in order to figure out which implementation of a particular d type you wanted to go to.
Instead of having one of these if statements, we were going to have a virtual object.
Because this is c plus plus and Siebel plus is all about objects and it's all about virtual methods.
So the concept was every tester had a type.
object associated with it.
By the way, the the term type still shows up in various parts of the code base, even though these type objects no longer exist.
But what the type object was was it had virtual methods for every single operation you can imagine doing on a tensor, adding, subtracting, sigmoid, whatever.
You name it, it was there.
And so every tensor would have a pointer to a type object that implemented all the things you wanted for the object in question.
And so to actually, you know, call and add on a tensor, you would instead go and the the implementation of the method on the tensor object were instead go call the add on the the type object attached to the tensor.
And that would, you know, do a virtual call to actually get to the real implementation in question.
Why did Zach do it this way? Because, you know, if you have done any object oriented programming, unreally normal way to design an object hierarchy in a situation is, oh, I got a tensor superclass and I'm going to inherit a float tensor from it and an intensor from it and so forth and so forth.
So there are a few reasons for this.
So one is that Zach really wanted Tensor to be what we think of as a pointer type.
So let's think about in in Python, if I have a tensor and I say y equals I I I have a tensor named x, and they say y equals x, then I want y to actually refer to the same memory and the same tensor really as x.
Right? I don't, like, make a copy in this situation.
We don't pass objects by value in Python.
They get passed by reference.
although some people would take offense to me calling it that way.
But in any case, you know, assignment and passing things parameters, they preserve object identity.
You don't create new versions of the object every time you do that.
In c plus plus, you have to actually, you know, say what you want your object to do.
So if you just define a tensor class with a bunch of fields for sizes, and strides and so forth, then if you pass that class by value to somewhere else, you will in fact copy all those fields.
when you get there.
And that's not at all what the Python semantics are.
So Tensor has to be some sort of class which doesn't do this.
And so we need to not we want a sensor to actually work like the python semantics.
And so you can't actually just subclass from Tensor directly because that just doesn't work at all.
Like, that's not how c plus plus causes work.
So another reason why Zac wanted a virtual dispatch rather than an if statement was because of the fact that CUDA support was this like sort of separate thing that was optional.
You didn't have to, you know, have a version of Pirate's with CUDA.
you could instead link against the dynamic library that provided CUDA support, and then that would actually let you, you know, get all of the CUDA functionality.
But you could also not link against that library and you'd only get the CPU support.
So you had these libraries living in two different dynamic libraries And if you've ever tried to write some code with multiple libraries, you might know that you can't actually call a function in another library unless you depend on that library.
And the way things were set up is the CUDA library dependent on a CPU library, but not vice versa.
So if you're in some CPU code, and you call this function and actually the tensor turns out to be a CUDA tensor, you need to figure out how to actually get to the CUDA library.
And the only way you really can do that is via via a virtual call.
The types provide the virtual call, they work pretty well.
It was pretty fast.
and we were happy for a while until the next thing came along.
So the next thing that came along was that, you know, we had this pretty cool a ten concept.
They're all these operators.
They all lived on the type object.
And some came in to us and they were like, hey, I want to define my own operator on top of the Tessa class and I'd like you know, like, I like you to find tons and tons of custom operators actually because I'm Facebook and I've got, you know, various very specialized use cases that I don't have a general purpose operator for, but I still wanna implement.
And this type class Right? With all these virtual methods on it, there's a problem.
You can't retroactively add more virtual methods to a class.
Okay.
Sure.
You can inherit from the class, but you can't actually But, like, you have to, like, inherit each time you do it and make sure you inherit from the thing you inherit from previously.
and this clearly is untenable if you've got you know twenty different people saying, hey, I wanna add my own extra operator in this situation.
And it was actually kind of important to make sure that people register directly inside the type object.
Because remember, we also have this feature in PyTorch called autograd.
And so actually, when you call a type object, you're not necessarily calling into the CPU or CUDA implementation In some situations, you might call to the autograd implementation that has something different, and then eventually you'll call into the CPU type afterwards.
So this need for open registration meant that it wasn't really tenable to keep using virtual tables.
Virtual tables are sort of Marvel of c plus plus design, but one of the reasons why they can be implemented the way they are implemented is because you're not allowed to add more methods onto them after the fact.
And we wanted to be able to load up extra libraries add new methods to them, and do that.
And this is when the dispatcher sort of in its modern incarnation came into being.
Right? So the idea behind the dispatcher is, okay, we are not going to we are not going to let c plus plus handle vtable layout for us is that we're gonna re implement the vtable ourselves.
And furthermore, instead of having all of the virtual methods for all operations laid out into a single table, in which case, like, it's not at all clear, like, how to add more things to the table.
we're just gonna maintain separate tables per operator so that when you call an operator, you know, you call add, you're like, okay, I'm gonna go look at the add dispatch table and it's gonna tell me how to go to CPU or CUDA because we want a lot of operators open registration operators, but for different back ends like CPU and CUDA, those get added way less frequently.
And yeah, and that sort of brings the dispatcher into sort of its, you know, a relatively modern form.
There are some things we added after the fact.
For example, we wanted the ability to do multiple dispatch So the the request for multiple dispatch came from a few places.
So one case where sort of we'd always known this was a bit of a problem, is we have support for sparse sensors in PyTorch.
And so you have this interesting problem, which is that if you've got a dense sensor and a sparse sensor, and you add them together, you wanna send this to the sparse kernel.
Because as far as kernels is gonna actually know how to deal with the sparse tester.
But in the initial implementation of the type objects that did dispatching, we always looked at the type of the first object to figure out where to go.
And since the first object's a dense sensor, we go to the dense implementation and then you have to do some extra tests to see if things are sparse and route them to the right right direction.
Multiple dispatch would let you change the behavior of dispatch for to respect the arguments of multiple to to respect the types of multiple tensor arguments.
if you had a dense and a sparse, okay, actually that means I should go do something else, not just, you know, blindly look at the first argument.
And Zach and I were talking about how to, like, implement multiple dispatch quickly.
during the fair off-site in Montréal, that's like a few years ago.
And Zach was like, hey, you know, here's how you could do it.
Right? You could maintain a a set of keys, a bit set of keys, representing all of the things represented by a tensor under some ordering saying which one you wanted to go to.
And then if you wanted to do multiple dispatch, all you needed to do was bit wise or all of these fields together and then just pick out what the left most bit on the resulting keywords to, like, get the, you know, highest party dispatch you want a dispatch in this case.
You didn't have to, like, do any, like, okay, looping over the arguments, looking for the right one.
It's just do this bit wise or extract out the first bit and and you're done.
And this basically served as the basis for the multiple dispatch implementation that is in PyTorch today, where you have a bunch of dispatch keys, they have a priority, and we always dispatch to the highest priority key.
these semantics came out because, you know, we had an idea about how to implement them efficiently.
Similarly, the work on automatic boxing came out of this problem, which is that, okay, you know, we have all this we have all these operators.
We made operators extensible And then we suddenly had a problem which is that we couldn't easily write code that was generic over all operators.
Previously, the way we did this was we had a cogeneration phase which, you know, knew about all the operators in PyTorch and was able to just write, you know, specialize people's plus code for each one.
But once we, like, open the gates up to let people register or whatever operators they wanted.
There were all these operators leaving outside of a repository, was the coach generation, knew nothing about and which you know we then couldn't really generically program in any reasonable way.
And so if the co gen doesn't know about it, Well, c plus plus does know what about the colonels in question.
And so Sebastian Messmer did this sort of years long project of sort of making sure that all objects, all functions even if they were registered outside of the dispatcher, could via templating magic actually be generically programmed over.
And so the the technology of back and fallback, which sort of only recently went to stable, is based on this.
So the dispatcher today is pretty complicated.
There's a lot of features that supports.
But, you know, if you sort of look through the history, you can see, you know, there were various design constraints that got us where we were today.
the design constraint of letting, you know, CPU and CUDA live in different dynamic libraries.
The design constraint of open registration, and even, you know, the design constraints of allowing for multiple dispatch or automatic boxing.
So these days, you know, the dispatcher has a lot of features.
You can do a lot of things with it.
And it's also a little slow.
Unfortunately, we've tried to make it fast but it's certainly a lot faster than if you were doing all of this in Python.
And I don't know.
The next time you have some project and you're wondering, oh, why is the dispatcher this way? Just think about the constraints.
It's a really useful way to reason about things.
Thank you all for listening.
See you all later.
.
