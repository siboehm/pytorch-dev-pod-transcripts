---
layout: post
title: "Native Functions Yaml"
date: 2021-05-25
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Native Functions Yaml

Hello, everyone, and welcome to the piketorch dev podcast.
Today, I wanna talk about native functions dot yammel, but I don't actually wanna talk about native functions.
at Yamal.
What I really wanna talk about is enough about our just in time compiler for people who are not compiler engineers and working on the eager portion of Pytorch.
You'll see what this has to do with native function dot yamal in a moment.
Okay.
So what is native functions dot yamal? Well, native functions dot yamal is this Yamal file named native functions, which basically describes every operator supported by PyTorch.
So imagine that you're thinking about sum or add or sub, each of these operators that Pytorch supports has an entry in this Yamal file.
And so this Yamal file basically is a sort of canonical source of information about these operators, except for a few exceptions, which we'll get to later.
Okay.
So why is there this Yamal file? Right? Like, if you were just writing a Python library, you'd expect, well, you know, if there's a bunch of functions that might library supports, I'll just write python definitions for them.
Or even, you know, if you're writing a library and you're doing c plus plus bindings, you'd expect, oh, well, you know, I'll have a bunch of c functions that implement the functions that I need, and I'll just register using them using pi by eleven.
So like, why do I need this separate representation? And as is always the case when there is an abstract representation about what operators you have, there's probably cogeneration lurking underneath.
And in particular native functions dot Yamal gets fed into a variety of different cogeneration pipelines which basically stamp out all of the boilerplate necessary to support all the things that you want to, you know, you want an operator to do.
And this is where, like, being jit for noncompiling engineers is important because we ask native functions that Yamal plays a very important role in generating our eager Pytorch bindings, that is to say, you know, the actual functions you call when you're just running Pytruch from Python.
But what it also does is it also generates bindings to torch script, our compiler and interpreter stack in high torch.
And so whenever you're like working on a new operator, when you're trying to define a new operator, Whatever it is you do also needs to work okay with compilers stack.
And here, it's helpful to know a little bit about what the compiler is trying to do with this information to, you know, figure out why, you know, there are certain constraints, about what kinds of things you can do in native functions dot channel.
So let's take one example to start.
So in indium function dot yamal, one of the things you do is you write down a so called schema string for any operator you wanna define.
So what does the schema stream look like? Well, let's take our example of addition.
So what is an addition? Well, it takes two tensors and it produces an output tensor.
And so the schema string for add basically is like, you know, tensor add open paren, tense yourself, comma, tense or other close per run.
Right? It it what it says is, hey, you know, here are the types of the arguments, here are the types of the outputs.
Pretty pretty standard stuff.
But if we look a little deeper into this type system, you know, the fact that we have this schema string, we the fact that we have this Jitskema format actually says something about what we are planning to target because In particular, the schema is not python types.
It's not c plus plus types.
It's jit schema types.
And what jit schema types represent is sort of the intersection of all language features that are supported by pipeline, as well as language features that are supported by c plus plus, and most importantly, language features that are supported by the transcript compiler.
So let's just take an example.
Right? So let's say that I wanted to write a a function in Pytorch that takes a void star pointer as its input.
Well, you can't do that.
And the reason you can't do that is while VoiceStar works as a, you know, type in c plus plus, there's no such type as void star in Python.
Well, unless you count, you know, one of the c type types, but we don't like most of the Python types that the Pytorch binding support are like stock types, like normal types, like integers, floats, bullions, tensors.
For example, So you can't write a function like that.
Right? And if you wanted to write a function that took voice star, you would first have to fix both the eager cogeneration code to understand a void star pointer.
As well as the c plus plus code, that would be easy because void star is very simple.
as well as the transcripts scope code to know how to represent a voice star pointer in what we call our box format or i value format.
which is basically a universal container for any type of, you know, object that you might actually pass into one of these functions in question.
So, yeah, there is a limited set of types available to native functions dot yamal, and this limitation makes it easy to actually, you know, write code that works for all of the platforms that we care about.
Of course, this can be kind of annoying sometimes.
For example, we don't have support for e noms in data function dot yamal because how e noms are defined in c plus plus and in python are fairly different and it's it's pretty involved.
There's no reason in principle we couldn't solve this.
But, you know, you have to actually pre declare an e nam in c plus plus, and you have to pre declare an e nam in python, except in python, that's not the python way to do e nams.
You just you know, provide a string saying what option you want.
So actually, most e nams are implemented sort of crappily using strings and I say it's crappily because like you don't actually wanna be passing around strings and doing string comparisons.
In Python, it's okay because string and turning happens.
And so if you're lucky, it's just a point of equality.
But in c plus plus, that doesn't happen.
And so you actually do want an e n m type we haven't implemented it yet.
Right? Because it's a little complicated to, like, work out a representation for e noms that works in all the situations.
By the way, if you're interested in doing this, Well, I'll talk to us because it is something that we've been wanting to fix for a really long time.
Okay.
So that's it about types and native functions about YAML.
What's another example of something that, you know, you need to worry about in native function dot yamal, not because it matters in eager mode, but because it matters in the compiler.
Well, great example of this is mutation and alias info in native functions dot Yamal.
Okay.
What's that? Well, if you ever look through the Yamal file, you might notice that there are some operations that have some little weird, Anna, like, extra, like fluff in their type signatures.
Right? So they don't just take a tensor as an argument.
they take a tensor open parenthesis a exclamation mark closed parenthesis.
What the heck does that mean? Well, what that means is that this argument isn't just being read in as a pure argument.
That is to say we're just taking it as an input.
We're also going to write to the argument in a situation.
So okay, you might say that's really useful for documentation purposes in eager mode, but like Why does it matter if I specify this correctly or not? Well, it matters because once again, we've got a compiler and our compiler wants to do certain optimizations and some optimizations might not be valid if you don't know if a operator is mutating its arguments or not.
For example, dead code elimination says that if I call a function on some operands and then I don't use the result, I can just get rid of that operation entirely, right, because it's dead code.
Well, I can't get rid of this function call, if the function is actually in the business of mutating the ten certain.
Because, you know, like, we might just be calling this function for the purpose of doing the side effect in question.
So it's actually really important to put down correct mutation information on your functions because if you don't, And then your function goes into the torus group compilot, which it will because the whole point of putting something in native functions dot yamal is so that you get all of this port right eager c plus plus script.
If you don't do it right, then your compiler may just miscompile your code.
It may you know, throw away your op calls, it may reorder them with other mutating op calls, that business all around.
Of course, what you really should do is just write your operator without having any mutation at all.
But, you know, sometimes that's not possible.
This is a really common mistake people make when they're defining custom operators.
because you're you're just like you just write a type signature down and you think, oh, this looks fine and the, you know, Pytrix accepted it.
What's wrong with it? All what's wrong with it is, you know, this downstream thing about the compiler.
So you're thinking about, like, what kinds of info the compiler needs That'll help you understand, like, what kinds of stuff native functions at yamal actually needs.
There's one more thing that, like, really, really, really affects people when they're making changes to native functions at Yamal.
And this is backwards and forwards compatibility with serialization formats and jit.
In the previous podcast, I talked about serialization sort of in a general sense, and I talked about this forward compatibility and backwards compatibility concept.
Well, this concept also applies to operator definitions.
So stepping back a moment, when we think about forwards and backwards compatibility in PyTorch, We usually only really care about backwards compatibility because you just write some Python program and you just want this Python program to keep working when you upgrade to the next version of PyTorch.
And there are a lot of changes that we can make to functions, which are actually backwards compatible.
For example, if we add a new keyword argument to an operator, but we give it a default.
From the perspective of Python, that's totally backwards compatible because Well, if I had a call to the if I had a call to the function before that didn't pass the argument, well, it'll just get defaulted.
And, you know, if I'm doing my job correctly, the default behavior will be compatible with whatever the old behavior was in that situation.
Well, well, but remember in functions dot yamal is being used in different situations.
And in particular, there are two particular situations where this is not exactly backwards compatible.
And by the way, these might be just mistakes, and we should fix these mistakes.
But sort of it's just how Pytros works today.
So situation one is For the longest time, when we serialize pipe so so stepping back a moment.
So one of the things that towards does is you have a model that has a bunch of function calls and we can serialize these function calls back into Python code.
And so something very interesting happens as a result of something that a compiler wants to do, which is whenever you serialize some functions, we actually write out all the defaults to the serialized model.
So let's just imagine like I'm doing a matrix multiply and I added an optional flag that says whether or not I should transpose the second argument or doesn't actually exist in PyTorch, but there are plenty of examples that are actually existing.
I just can't think of them right now.
So in this situation, if I write a, say, Matt Moll a b.
What will actually get serialized is Matt Moll a b True, where True's a sorry, false.
where false says don't transpose the second argument.
That's kind of weird.
Why does the jit do that? Well, one of the reasons the jit does this is You know, one of the things that it does when I'm compiling your program is it tries to translate it into an intermediate representation that's easier for the compiler to deal with.
And one of the things that makes i r s easy to deal with is when they are very regular.
So what do I mean by regularity? Well, it means that I don't have to like you know, go ahead and canonicalize stuff every time I look at it, I can just assume that things are in canonical form.
And an example of something in canonical form is a function call which has all the defaults actually explicitly written out as opposed to like implicit.
Because if they're implicit, you have to go figure out, you know, what the behavior, what the defaults are and fill them in if you wanted to, like, actually write code that operated on the semantics of this function.
Okay.
So because this IR representation transformation happens, well, as an accident, when we re serialize things out, we actually just lost the information about whether or not, you know, something was explicitly defaulted or not explicitly defaulted.
And so we just always serialized it out.
Why is this problematic? Well, it's problematic for Ford's compatibility.
Recall from the previous podcast, Ford's compatibility refers to if I serialize a model from a newer version of PyTorch and let's say that it doesn't actually use any of the new features of PyTorch, which, you know, would necessitate using the new version of Pytorch, can I run it on an old version of Pytorch? And so if you add this defaulted new parameter and while it's getting serialized out, Uh-oh.
When you, you know, try to load this model in an old version, there will be this extra parameter that your old version of Pytruich doesn't understand.
and well, sucks to be you, the model can't be loaded anymore.
So there is a way to solve this problem in PyTRISH master, and I don't exactly remember how we resolved it.
There it's either some sort of like backwards compatibility sorry, forwards compatibility Well, one is we don't really offer Ford's compatibility, but I think there's some, like, surgery you can do to fix the problem.
Or it might just be that we fix this problem to begin with.
But, like, the the meta point here is that this was a problem for a while.
And the reason it was a problem is because you know, jit is using this representation in a way that is different than how you normally might conceptualize it in just eager a mode.
And so to just understand the consequences various changes you might make.
You have to also understand, you know, what's going on in shit.
Is this bad? Like, what if we, like, just wrote our format really, really nicely and explain all of the invariance in question and like you could just read up about them and know everything.
Well, yes, ideally, this would be the case.
Ideally, we would have a really good backwards compatibility and forwards compatibility story, and we wouldn't have problems like this.
Great.
If you want to work on this, you know, come talk to us like, you know, this is this is a really important project for Pytorch and And we've just, you know, been very slow in actually getting because who wants to work on factors compatibility, honestly? I do actually.
but I'm I'm always working on other stuff, unfortunately.
So yeah.
So so what did I talk about today? Right? So I talked about native functions dot yamal.
I didn't really tell you, you know, how to write things in native functions dot yamal, and I don't really want to in this podcast because there are pretty nice documentation that you can look at.
What I wanted to go over today was more, you know, why does native function dot Yamal have all of this stuff? And the reason it has all this stuff is because, well, there's this compiler stack attached to it.
And, you know, there are a bunch of constraints that, you know, we need to solve simultaneously in both cases.
So if you ever find yourself wondering, you know, why is something this way? Well, maybe there's something in the compiler that needs it to be that way.
And also, I also wanna emphasize that compilers are not that magical.
Like, there's not that much they're doing.
So you can't understand it even if, you know, you don't work on the compiler on day to day.
And like, once you understand it, you might actually be able to look at the situation and say, hey, actually, there is no reason for it to be this way.
And we can fix it.
and then, you know, you can just fix it.
And that's that's pretty powerful.
And so a generalizable lesson that applies to all of software engineering.
That's everything I wanted to say today.
Talk to you next time.
.
