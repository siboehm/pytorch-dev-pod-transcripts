---
layout: post
title: "Functionalization"
date: 2021-05-06
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Functionalization

Hi.
My name is Edward, and welcome to today's edition of the Pytorch dev podcast.
Today, I want to talk about a process called functionalization which is used in multiple parts in the byte rich code base.
What do I mean by functionalization? Well, I don't necessarily mean the conversion of things into functions, But what I actually mean is the removal of mutation from operations that you do in PyTorch.
And it turns out that, you know, being able to remove mutation, being able to transform an otherwise mutable program or trace into a purely functional form is a very useful transformation and one that we use in several places in PIKTRA.
So I just wanna talk a little bit about why this is useful.
and then tell you about how we do it.
Okay.
So why is functionalization important? Well, a long long time ago in our, like, pre pirator zero point four days, we didn't actually support doing autograd and mutation at the same time.
And there was a reason why this was the case.
It's because, you know, when you have a program and you just, you know, write a bunch of pure function calls, you can easily just, you know, create a autogard graph that represents the calls you just made and then replay that graph when you go backwards in time.
But if mutation is allowed in the mix, if you're allowed to sort of modify something in place when you when you are working on the forward pass of your function, not only do you have to somehow deal with a mutation, but you also have to somehow modify all of the other aliases, all the other views on the object in that situation.
And this is kind of complicated and difficult to think about how to actually do this.
And so the way that, you know, we actually implement this in Pytorch is morally inside Pytorch's internals, we convert your program into a functional form, one where the mutations are removed And so the autograph trace is not, you know, recording, hey, you know, this is the mutation that happened, but actually, here is the purely functional version of the program that would actually give you the, you know, same computation that you would have gotten if you had done the mutation in question.
So Why is functionalization important? It's important because we can use it to implement automatic differentiation in the presence of mutation.
You don't have to do this But one of the things people really like about using Pytorch is you can just sort of do all the thing that Python normally lets you do.
And one of those things is mutate tensors.
So it's kinda nice that, you know, Autograph works with this.
Another thing that this is useful for and got repurposed after the fact is Pytorch has an integration with XLA.
XLA is the back end for TensorFlow.
You know, it's a very nice back end, generates good code.
And there's something very important about it, which is it is purely functional.
It doesn't support mutation.
And so when we have a Pytrich program that has a bunch of mutations in it, when we translate it into XLA HLO IR, we need to figure out a way to get rid of all of those mutations.
And so in fact, the torch x lay extension developed by David Labenset and Co actually does, you know, the same kind of functionalization that our autograd pass does when mutations happen in our program.
So it's useful in a bunch of places.
In the past, we sort of like, you know, reimplemented this trick as needed, but we're gonna eventually work on adding functionalization as a proper pass to PyTorch.
So anyone can opt into it.
if it's something you need for your back end.
Okay.
So I've talked a little bit about functionalization and why it's important, but like, you know, why is this a hard thing to do? because if you, you know, ask a, you know, die hard functional programmer, oh, they'll just tell you, hey, you know, getting me to mute.
bit of mutation is not too hard.
You know, instead of like adding to to a variable, you directly to a variable, you just say, okay.
Well, you know, x plus two equals y and then anywhere you previously refer to x, you just refer to y instead.
So what's the big deal? And the big problem is aliases.
So let's say that I have a tensor and I, you know, take out a bunch of views on it.
and then I fill the tensor with ones.
The modification that I did is not just, you know, take this tensor and replace it with an entire tensor full of ones.
It's also all the views that I've taken of this sensor, all those views also need to get filled with once.
And this poses a very hard implementation challenge for us Because when I am, you know, riding a runtime system for Pytorch, we're we're doing a reference counted implementation we want things to get promptly disposed of.
And so this object that I'm filling all with ones doesn't actually know where all the views are.
So so imagine that, like, for any given sensor, you knew all the aliases to that sensor.
then you could still functionalize in, you know, a little bit complicated but not too bad of a way.
And the way you would do it is you would say, hey, here's my tensor.
Here are all the aliases.
I do some mutation to the tensor and then I look up all of the aliases and then I replay that same mutation on each of the aliases.
Well, okay.
I I had to, like, narrow the scope of the mutation.
Right? Because the view is only looking at a part of the sensor.
and so I just only need to apply the mutation from that part.
But then I just go ahead and apply the mutation to each of them.
And in the same way, you know, let y equal x plus two and then, you know, all previous references to x or not y, I can just update all of these one by one and then I have a new updated functional graph.
that doesn't have any reference to mutation.
But I can't do this.
I can't actually maintain this list of aliases because if I did maintain that list of illiuses.
Well, one is were ref kind of so if they were strong references, then you'd keep all of your views live even if, you know, no one was actually using them.
Right? Like, if if someone takes out a view to your tensor and then you mutate that tensor and that view never gets used you need that view to go dead in that situation.
And if you made these all week references, well, that still causes problems because you have to, you know, do pile of bookkeeping on the tensor in order to keep track of all these views.
You no longer have a fixed size representation for a tensor.
The set of aliases to it may grow unboundedly.
I actually remember a long time ago when Sam Gross was initially implementing our c plus plus autograd system, and he was trying to get mutation to work in this situation.
He came to my desk and he asked me, hey, Edward.
So I'm trying to figure out how to, you know, deal with these mutable aliases.
And, you know, I was thinking, you know, could I just store the aliases for all the tenses and update them? And I was like, Sam, that's not a good idea.
Don't do it that way.
And so Sam went away and he thought about the problem and he came up with a better solution.
And I wanna tell you how that solution works today.
So just to recap the situation.
Right? So we wanna do a mutation to a tensor and we want to somehow get all of the aliases, all of the views on that tester to see the change in question.
But we don't know what those views are.
So how can we make sure we actually get this mutation, get the knowledge of this change to all those sites? Well, the answer is, you know, If you can't do it now, do it later.
So when we do a mutation to some base tensor, we say, okay.
Here's the mutation that has happened, and we also flip some bit, flip some version saying, hey, everyone else, all of y'all y'all views If someone else tries to ask you what the functional computation graph corresponding to you are, and it turns out the base has changed under you, in the meantime, from the last time you came and look at us, you need to stop and re compute what your new value is subject to the mutations that happened.
So let's just go through an example.
Let's see what happens here.
So let's say I've got my tensor a.
I have a view, v, on the on the sensor a and I add two to every element in a.
So I go ahead and do that I update the version on a, so it says, hey, I'm out of date.
V, you know, when it was taken out from a, recorded the old version.
So I got version zero.
Version zero recorded in v.
I update the version on a to go to one.
And so a records, hey, this is the mutation that I made in the situation.
And so the next time I access v, the first thing that I need to do is I say, hey, v, are you up to date? And v goes and looks at its base, which is a, and it says, last time I looked at a, I was version zero.
But now, a is version one.
So I need to do an update.
So v then goes and looks at what the changes that were made to a.
In the meantime, we're reapplies them to v and then says, okay, here's the up to date representation in purely functional form.
of the contents of v.
Now in Autograph, we don't quite replay things because when we have the when we have the computation represented by a, we actually don't have to, like, you know, replay the mutation on v.
We can just say, okay, just take whatever the current state of a is.
Take whatever the actual condens of AR, you know, and the functional trace that created it, and then just reapply that view operation.
to actually get the contents of v.
Right? So it's sort of not not if you imagine, like, two tracks.
Right? running and v is one track and a is one track.
It's not like a is making changes and those changes get merged into v one by one, but actually that every time you make a change to a and then we look at v, a new branch branches off a and you just sort of forget all about the old branch that you had before.
In fact, Autograph even doesn't further optimization, which is we don't even have to remember what the views are.
because every view is related to its base sensor simply by the strides that are recorded in the view.
If you have read my blog post, which is an introduction to Pytorch, I explained what Strides are.
And, you know, any view operation boils down to a re striding at some offset on the tensor.
So we just have this as strided backwards that just gets applied in this situation.
Of course, XLA doesn't actually support striding.
So for XLA, we actually just replay the view operations, and that's how it goes about doing these updates.
So that's basically how functionalization works.
So we don't eerily update all the aliases when we do a mutation.
and said that we lazily update them when they get accessed.
This preserves the rough coding properties we want where we only ever have references from from, you know, subsidiary things to the computation graphs that preceded them, and we don't need to maintain lists of tensors that tell us what the aliases of a base tester are.
So Another pretty interesting property about this scheme is it's actually quite a bit better than static analysis.
So, like like, let's imagine that your LLVM or some sort of compiler or like the torch scoop compiler and you want to you have a program and it's got some mutation in it.
you wanna remove that mutation because maybe you've got some functional optimizations that work better in the situation.
Well, when you're in the compiler setting, it's actually kind of difficult to remove all of the mutations because you just don't know what the aliasing properties of your inputs are.
This is why actually, like, when you're writing functions, sometimes putting a restrict qualifier which says, hey, this pointer input is guaranteed not to alias with this other pointer input.
the restricted qualifier is so important because the fact that you can prove that they're not alias because you told the compiler that then enables a bunch of optimizations that the compiler can do.
But in general, the compiler has to be very conservative and it has to, like, sort of, you know, you know, if it doesn't know, it has to assume, oh, this could alias with something else, and that just impedes a huge number of optimizations you might do.
whereas PyTorch, which is sort of just running this functionalization as we run our program eagerly, always has absolutely precise alias information about what exactly alias was with something else.
And so we can absolutely perfectly remove mutation in the situation without any loss of fidelity.
Of course, you know, like, this is only for a single trace, whereas, you know, your optimizer might be working under, you know, very different situations where some things may alias sometimes and some things don't.
Right? So it's the price of generality.
When we specialize specialized specialized to the specific case, we can do something really good in the situation.
So that's it for functionalization in Pytorch.
It is how we, you know, sometimes I like to tell people, hey, you know, Pytorch kind of wore a hair shirt where we were like, hey, we care about mutation.
We care about supporting in place operations.
And then we had to do a whole bunch of, you know, complexity.
Like, we actually have to work pretty hard to make sure mutation works for our users.
But at the end of the day, like, how do we do this? We I think we we map the mutable operations into the functional universe and then we do the things that, you know, automatic differentiation, all that good stuff.
So it's actually pretty nicely factored in this way.
And this is one of, like, really joyful things about working, about Pytorch.
Alright.
That's all I have to say today.
Talk to you all next time.
.
