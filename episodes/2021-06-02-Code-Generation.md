---
layout: post
title: "Code Generation"
date: 2021-06-02
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Code Generation

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk about code generation in Pytorch.
Code generation refers to the practice of writing other scripts that generate code for you.
In the case of PyTorch, these are our Python scripts, which get run as part of the build process and produce a lot of c plus plus files that actually make it possible to build PyTorch as a whole.
Code generation is it's kind of the heavy guns.
Right? Because when you start co generating your code base, a lot of things stop working.
For example, if you've got an IDE and you wanna jump to definition, Well, whoops, looks like your, you know, method you're looking for is actually in a cogenerative file, which means it's not in your working directory, and you have to go build PyTorch first before you can actually go look at it.
So it makes things kinda confusing and, you know, most of the time, if you're just writing a c plus plus project, you try very hard not to do cogeneration.
But in the case of Pytorch, we've used cogener actually from the very beginning of the project.
even back in the days when eight ten wasn't even a thing.
And it ended up being a pretty good trade off for us in terms of what it allows us to do.
The high level of why, you know, cogeneration tends to be a good idea is that it lets you greatly reduce the amount of code you have to write manually.
in a project.
If you had, you know, a, you know, hundreds of classes that you would have had to have written, you know, one by one Well, if you have a code generation pass, you can just generate them all from, you know, a few lines of Yamal, and you don't have to worry about it.
So that's what it's doing in PyTorch.
Code generation is being used to generate a lot of code that we otherwise have to write hand by hand.
It makes the framework more maintainable but, you know, it is kinda complicated.
And so I just wanna talk a little bit more about what kind of stuff we're using cogeneration four.
Also, what are some of the pros and cons of using cogeneration and some other counterpoints in the design space because cogeneration isn't the only way to skin the cat.
necessarily in some situations.
Okay.
So what are we using cogeneration for in PyTorch? There's a lot of things that we're doing using it for, and at a high level, the the biggest way to think about, you know, why we're using cogen for any given thing is because usually, it was something that we needed that you can't do with plain old fashioned c plus plus method programming with templates.
So a really simple example of, you know, c plus plus just doesn't support enough language features to do this, is generation of APIs, like functions or method on methods on classes, you know, based on a small amount of data.
So for example, we have a type named Tensor, and it supports a lot of methods on it.
And those methods essentially call into another class, it's really a dispatch mechanism that, you know, is very uniform.
So, like, for every method, what it does is it just, you know, takes its arguments and calls into another function that, like, actually does does the method processing for us.
And in one of the, you know, philosophies in the c plus plus API and PyTorch, is that, you know, we want it to be possible to to just write the same code you would have written in Python.
So if you wrote x dot add in Python, You can write x dot add in c plus plus.
But c plus plus doesn't have operator dot overloading, so we have to actually manually write out every method by hand whenever we wanna write a class like Tensor, which supports a method like this.
So we don't write out these methods by hand because we have hundreds of methods on the Tensor Class is that we use code generation to actually do this.
Another example of us using code generation is when we do automatic differentiation, see my previous podcast, we need to generate a class representing the set of save data for any given piece of autograph information.
And we actually generate one class per piece of autograph per operator because autograd might save different things depending on the operator in question.
Right? Because there might be different mathematical values from the inputs that you need to compute the derivative in these cases.
We don't do a box representation for autograd because that would be less efficient instead we just have a specialized class for each operator that only contains fields for exactly what we need.
And oh no, Once again, there's no way to in c plus plus conveniently generate a ton of classes with slightly different fields based on some simple specification of what the things are.
So instead of having to write them all up by hand, we also use cogeneration to generate this.
cogeneration is also used in some cases to deal with things that don't live in c plus plus at all.
For example, we have a bunch of Python bindings.
We do co generate the arc parsing logic for parsing the arguments from them, but we also need to generate p i i stubs, type stubs that make the type information available for all the c by an England question.
Well, how do we do that? Well, there's thousands of operators.
So once again, we co generate the Pi'i.
So someone we we didn't use to have this capability.
We we didn't have any type stubs for it.
And all someone had to do was just go and write an extra python script that knew how to generate these python's type stubs.
and that was it.
They didn't have to, like, painstakingly go through every operator in Pytorch and figure out what their type signature would be and then saddle us with the burden of having to continuously maintain this extra set of stubs.
Instead, it just gets generated by code in this situation.
some of the time, what we do is we say, okay, you you wanna implement an operator and you need to implement a CPU and CUDA version of this operator.
And usually, there's a fixed prototype that we expect a user to implement in this situation.
So we also use cogenerate to generation to generate the prototypes for these functions so that, you know, you know what you need to implement downstream.
Okay.
So those are some of the main uses for cogeneration inside PyTorch.
So what are the benefits of using cogeneration? As I said, I've harped on repeatedly about, you know, often we use cogeneration when there's no other We just can't do what code generation wants to do using just c plus plus templates or other mechanisms.
But there's also other reasons why code generation is something that you know, we reach for.
For one, when we build a cogeneration system in Python, we can actually do much more complicated things with surface syntax For example, we have a native functions dot yamal.
We inside it, we have this miniature domain specific language for specifying Jitskema, which is like something that we have to write a parser for.
And, you know, we also have derivatives dot yamal, which is this compact representation of writing derivatives for functions.
And yes, in principle, you can write a templated piece of code that is a parser for some arbitrary syntax.
and people have done this just to show that it can be done in c plus plus.
But in general, c plus plus is much better at, like, modeling meta programming based on, like, C plus plus types.
Right? Like, that's how, you know, partial specialization and tricks like that work.
So C plus plus really compact code when you, like, wanna look at the type structure, of your school's programs and met a program off of that.
Really bad, horrible, awful, no good looking code when you want to, like, implement a parser that happens entirely at compile time.
And yes, constexpr makes things better and the, you know, bigger your c plus plus version is that all it makes it better.
But unfortunately, PyTorch is still stuck on c plus plus fourteen.
Hopefully, we'll get to c plus plus seventeen soon.
But, you know, we need to work in a lot of different platforms and that sort of puts a limit on how futuristic our c plus plus code can be.
Another reason that we like using Python code generation is it makes it easier to write better error messages.
template error messages in c plus plus are famously horrible.
Right? Maybe if we get c plus plus concepts, in the future, things will get better.
But, like, you know, a lot of people don't really know how to debug c plus plus template errors, but they're perfectly fine if, you know, it's just to Python script, and there's, you know, oh, I'll be a complicated Python script, but it's, you know, raising as exceptions somewhere.
because then you can add print statements you can, like, look at, you know, what you can tweak tweak things around, you can print extra things out, and it's just easier to, you know, deal with than c plus plus.
Yes.
you can figure out how to do all of these things in c plus plus.
But c plus plus meta programming debugging is a skill, and most people don't have this skill.
whereas most people do.
And when I say most people, I mean, like, you know, most developers on Pytorch.
Most developers on Pytorch do know how to write Python code do know how to debug Python code, so that makes things a lot easier.
A sort of similar thing related to this is that in c plus plus templates, you often have to do very complicated encoding mechanisms to, like, represent complicated data structures because, you know, like, as I said, Salesforce is all about, like, operating on types.
And if you actually wanna do data, well, you have to work pretty hard.
And in Python, well, you can just write a data class and, you know, use that to represent whatever data you need to pass around.
In fact, our cogeneration is very strongly typed Python.
We use data classes everywhere, frozen data classes.
and we it's fully type annotated with my pie.
And that makes it easy to also do refactors where you just, you know, make a change to the data type and then you just look for all the places you need to update in a situation.
One last thing.
With a cogeneration framework, we generate c plus plus which then is compiled by the c plus plus compiler, which means that if something isn't working, you can look at the generated code and be like, is this the code that it would written by hand? And so it's just generally easier to reason about the performance characteristics of Python based code generation because you're often trying to generate code that looks like code that you would have written by hand.
And with templates, it can be obscured because your there's this level of in direction.
You're never actually looking at the code that actually gets generated and it's easy to accidentally put in inefficiencies when you write things that way.
I spend this all time like saying with the pros of doing cogeneration art, but, like, they're also some very big cons.
Right? So I've talked about a few of them already, such as that cogeneration is complicated Lot of people don't really wanna, like, deal with this random Python script that is generating code.
If you do a bad job at maintaining your Python code that generates c plus plus, can be really, really hard to maintain.
And in fact, that was the state of the old code generation before we we wrote it again with strong types.
But there's some less obvious cons to cogeneration as well.
One is that cogeneration is not portable.
What do I mean by that? What I mean is that, let's say that, you know, you have some stuff that generates code for you, and then you have some external user of Pytorch that also wants to make use of this code generation pipeline.
If I had a c plus plus template, I could just say, oh, instantiate the c plus plus template in your project and then you can get whatever functionality the c plus plus template gave you.
And they don't have to do anything extra in their situation.
Whereas if I have a Python code gen script, Well, now I have to like actually design the coaching skill to be rungable outside of Pytorch for some, you know, extra data that the user does in question.
And it's just there's a lot more work you have to do to make sure something is publicly available.
We are doing some of this work actually.
So for external back ends, we spent a long time giving only a c plus plus template based API for registering extensions, but it eventually became clear to us that that was just wasn't enough.
We didn't have enough features to do it.
And Brian Hirsch has been working on out of three co gen for back end extenders.
It's pretty cool.
I'll post a link to it in the podcast description.
And but like, you know, we spent a long time not doing this because well, there's a lot of work you have to do to actually make external cogen work.
And I just wanna talk a little bit more about, you know, I I said previously that c plus plus templates are pretty good for doing meta programming based on the c plus plus type system.
Right? And it makes sense because it's built into the c plus plus compiler, which knows all the vagaries of how c plus plus types work.
And it has turned out that when we write a Python cogeneration framework, we actually need a, like, you know, model of the c plus plus type system because sometimes we just need to do administrative stuff like conversion from one type to another.
And while, you know, the best way to do that is to actually know something about c plus plus types so that you can, like, you know, basically run the whatever implicit conversions or tight matching that c plus plus would have done in in this situation.
So we had to implement that.
We have a crappy version of the c plus plus type system.
and our code gen, it would have been easier to do this in c plus plus itself sometimes, perhaps.
Because sometimes it's very easy, but, you know, when you add a little extra feature, then it becomes difficult to do something with templates.
So I spent most of this podcast being like, hey, you know, you can either do code generation or you can do c plus plus templates.
And these are two points in the design space for doing this kind of thing.
And one of the reasons why I put these as the two, like, possibilities is because both of these have the same efficiency characteristics assuming you've done it correctly.
Right? C plus plus templates get instantiated every time you give them some parameters so they can generate code that's just as efficient.
as if you would written it by hand, which is what, you know, a cogeneration would do.
But there's actually a third point in the design space, namely boxed fallbacks.
So what are box fallbacks? Box fallbacks are basically a way of writing polymorphic code that runs at runtime rather than at compile time.
And the way this is done is by making sure all of the inputs to a operation in question are boxed.
They they they're stored in a uniform representation called an i value, and then you can actually write c plus plus code that's polymorphic in that sense.
By the way, if you're used to be able to doing generic programming, say, in Python, or in Java where you just, you know, like, write some use, like, a reflection API or something like that to write code that works no matter what the types of inputs are.
You know, that you're also taking advantage of the fact that those languages, their internal data representations, are all box, they're all uniform, so you can just write runtime code that does this.
C plus plus doesn't have that.
So we have to actually turn things into their box representations before we can write this uniform code.
Boxfallback code is often way simpler too.
Right? I recently Brian, once again, he's been working in the space, so he's the expert.
Brian has been, you know, taking some code that we used to do in Cogen and writing it using a box fallback, namely some CPU fallback code.
So what does this do? It just says, hey, I wanna run an operation, but I don't have it implemented for XLA.
So I'm gonna pass it to CPU and then run the operation on CPU and then put it back in XLA.
And it's really like easy to do the box fallback version.
You just do the obvious thing.
You, you know, iterate over the arguments.
You look for ones that are x lite tensors, convert them to CPUs, call the actual thing and then, you know, iterate over the results and turn them back into x l a.
Very, very simple.
You'd have to do quite a lot of work to, like, write the cogeneration version of it.
And you probably have to do less work, although still some amount of work to write the c plus plus template version.
The box fallback is very simple.
It's easy to debug as well because you can add print statements in the normal way.
There's no templates involved.
The problem is it's less efficient.
Right? Because you're boxing things up and you've got this little interpreter that, you know, has to go and look at what the types of everything are.
So box fallbacks simple, and, you know, they work at run time, so they, like, work even when you can't see the code in question that you might need to do, but it's less efficient.
So you probably only want to use them in cases where efficiency is important.
And CPU fallback is definitely one of those cases because well, you're falling back to CPU.
So, like, you don't expect it to be fast.
You're just trying to make it work at all in in the first place.
So that's a list of everything that I wanted to say about cogeneration.
One of the open questions that I have as a programming languages person is Is there a way for us to have the best of both worlds? Right? So I I I had this picture of, oh, I can meta program things ahead of time and it's kind of complicated, but it's really efficient, or I can write this interpreter that does everything at runtime.
It's simple to write the less efficient.
Can I have the best of birth wells, for example, by writing an interpreter and then partially evaluating it so that I can get the fast compile time version? Well, I can't easily do this if I write my interpreter in c plus plus, but maybe if I write it in a different language, it'll be easier to do.
That's something that I've kind of been thinking about, although we don't really have any concrete projects for dealing with this.
That's everything I wanted to say for today.
Talk to you next time.
.
