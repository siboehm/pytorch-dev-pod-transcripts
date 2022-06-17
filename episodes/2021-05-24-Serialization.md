---
layout: post
title: "Serialization"
date: 2021-05-24
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Serialization

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk about a somewhat dry, but still very important topic to Pytorch, namely serialization.
serialization is the mechanism by which when you have a cartridge program and you have some tenses floating around or God forbid a more complicated program such as a cartridge module or a transcript module, it allows you to serialize this data to disc so that you can load it up again, you know, when you do another run.
So in any sort of, you know, usual patronage program, you are probably making heavy use of serialization because you're doing things like, you know, doing your training loop and then saving your trained weights to disc.
So, you know, because, like, you actually wanna use them later for something, for example.
Okay.
So how does serialization work? Well, it's a long story.
So I think the way easiest way to understand where Pytronch is with serialization is today, is we're gonna first talk about how serialization works in general in Python.
And then we'll talk about how historically Pytorch did serialization based off of this, and then we'll talk about the new developments, namely Jit zip file based serialization, which is what more recent versions of PyTorch are using by default when you do torch dot save.
Alright.
Ready? Let's dive in.
So Instead of answering the question, how does Pytorch do serialization? Let's ask a easier question, which is how do you do object serialization in Python? And the answer to this is, well, there are a bunch of ways to do things, but there's one that is very popular and a lot of people use, namely pickle.
Pickle is a protocol and file format for doing arbitrary Python object serialization in Python.
So, like, if you have some class, so you've got some object, you've got some numbers, you've got a list, whatever, you can run it through pickle and pickle will give you a byte stream that you can put on disk and then you can unpickle things later.
How does pickle work? Well, pickle is find for on a per object basis, and the way you define how a pickle works is you define what's called a reduce magic method.
underscore underscore reduce underscore underscore or, you know, if you're cool, it's actually underscore underscore reduce underscore e x underscore or the EIX meaning that you also get to know what the pickle version is.
So so, you know, for any given class, if you wanna be able to serialize something using pickle, you just define what the what the pickle sorry.
What the reduce function for things should be.
And the way you write one of these reduced functions is actually it's sort of recursive.
You just define the serialization you want in terms of smaller and more primitive objects that you wanna serialize.
So let's imagine that we're serializing a tensor.
So the way we serialize a tensor is that we actually return a twofold from our reduced function And you can actually go look at this code inside torch slash underscore tensor dot pie.
It's it's all implemented in Python, at least one version of it is.
So what do we do? Well, you know, we get our tensor and we're gonna construct a twofold containing all of the important pieces that we need to rebuild the tensor.
So it's gonna contain our storage, It's gonna center our store our sizes.
It's gonna store our strides.
And very importantly, it's also going to store a function that says how to take all of these, you know, particular pieces to that are there, namely, you know, the size and the stride and the storage, and reconstitute this into some actual tensor.
Because when we actually want to, you know, load our tensor from pickle, like pickle needs to know how to actually take one of these, you know, two poles and turn it into the actual object in question.
So you do that by providing a function a rebuild function as we call it internally that takes the various pieces that you serialize one by one and reassembles them back into the hole.
So that's how pickling works in general.
And pickle itself is a pretty simple file format.
there there are plenty of tools you can use to look at pickle.
So if you've like ever thought, oh, you know, pickles are just these opaque blobs that I can't actually look into, well, okay, they are binary objects, but like the format is actually very simple.
It's like a little stack machine that you use to just build up various data structures inside pickle.
By the way, if people have ever told you that, like, pickling and unpickling arbitrary objects is unsafe, that's because, you know, pickling can induce object construction And so if you're not safe about what objects you construct via pickling, then you could accidentally trigger, you know, remote code execution.
Like, say you unpickle an object that actually goes ahead and, you know, runs a shell command from whatever you passes the constructor.
So that's something to keep in mind for as we're getting later in this podcast.
Okay.
So Python serialization usually done using pickle because pickle is built into the standard library.
Everyone sort of knows about it.
It's got a protocol for defining this that most people do.
It's actually a little tricky to, like, pickle things correctly.
For example, imagine that you are pickling an object And and you you so you have a class and you are pickling it.
And then in a new version of your software, you add a new attribute to your class.
Right? So adding a new attribute ordinarily is a backwards compatible change because, well, you know, like, all the old users of your code weren't using that attribute So what what skin off their back is it if there's a new attribute.
But with pickling in the mix, this is actually usually a BC breaking change.
because any old pickles from older versions of your class don't actually have this attribute set.
So when you actually write the unpickling code for your code, you'll unpickling object that is missing this attribute.
And so if you don't like, if you assume that the attribute is set, which is very reasonable thing when you're writing class, it'll break when you unpickle this old thing.
Fortunately, Python has another mechanism for overriding behavior in this situation.
There's a magic method called set state, which gets called whenever you're, you know, you're actually populating the state quote unquote, for an object that's being pickled and that doesn't actually have a full on reduce implementation.
And so in that situation, way to make things b c is usually just to look for missing attributes and fill them in before you load the object.
Okay.
Long long tangent aside.
Okay.
So how does Tensor pickling, historically, how was it implemented? Well, we did the same thing that everyone else does and we used pickle to do it.
So what do you expect to see? Well, you expect to see on Tensor, there is a reduced implementation, and indeed, there is a reduced implementation in torch slash underscore tensor dot pi, and it has a bunch of functions.
For example, it has a function that knows how to rebuild the tensor.
Actually, this function is called v two because in zero dot four when Sam goes merge to Tensorid variable, he actually, you know, changed also the serialization format in a backwards and forwards incompatible way.
So, you know, we had to make a new rebuild implementation.
Digression about forwards and backwards compatibility.
So backwards compatibility typically is this idea that if you serialize a tensor to, you know, some save format.
A backwards compatible software means that when there's a new version of the software, you can load that old version of the pickle in a new version.
Backwards compatibility is a good idea.
We try very hard not to break backwards compatibility ever, especially with the serialization format.
However, there's a very similar and also important notion called forwards compatibility.
So four's compatibility means if I serialize an object from a newer version of Pytorch, can I load it from old versions of Pytorch? And, you know, it should be clear to see that, like, maintaining indefinite forwards in cat incompatibility means you can never ever change the serialization format.
but it is useful to be able to, like, load new tinctures from older versions.
So whenever we've broken forward's compatibility, we've usually had some mechanism by which you can get back the old format.
So if if you're just in a pinch, you need to send something to an older version of PyTorch.
Okay.
So digression over.
So we have we're on v two of the tensor serialization.
And, you know, v two is obviously not force compatible with v one, but you'd have to be running PyTorch like zero dot four.
So that's like ancient history.
and no one really cares.
But for zip file format, this is gonna be relevant in a moment.
Okay.
So what do we do for Tensor? Well, we do exactly this.
So, you know, we've got function to rebuild tensors based on the data.
What is the data? It consists of storage, sizes, etcetera.
Storage itself also has an implementation of how serialization works, namely, you know, you just you just there there's a reduced implementation But this reduced implementation does something very interesting, which is it actually calls into torch dot save to do the implementation.
And so now here is sort of the first interesting thing that's going on, which is that actually pickle i.
e.
the interface that Python gives you for pickling objects is not the same thing as torch dot save and torch dot load, which is the other sort of very published mechanism for doing serialization in PyTorch.
Like, usually, like, when you look at the tutorial, you don't, like, directly instantiate a pickle object and pickle Python objects.
you actually use torch dot save to save these things.
So what does torch dot save do differently? Well, if you go look at the implementation of this file, also all in Python, easy to look at.
What you'll find is that we actually do most of the things you'd expect, right, which is that we are gonna create a pickler, and then we're gonna feed it the data in question, and then out is gonna pop a byte string.
What we do differently is that we want to deduplicate storage that are shared between multiple tensors.
So let's imagine you're serializing a list of tensors and you have a, you know, law the list of tensors actually list of views onto a single tensor.
So if you, you know, serialize this naive way, we you would, you know, stamp out a copy of the same data for every single occurrence.
And and then once you de serialized it, like, these would all be different tensors.
And if you mutated one of them, the other tensors wouldn't get mutated.
That's bad.
We don't want that.
We want this the sharing to be preserved during pickling.
And so the way this is done is we use this other mechanism in Python object serialization called persistent IDs.
where basically for any given object that's being pickled, you can override the behavior for what happens in that situation.
And so when we see a storage, we actually record a persistent ID that records what that storage is.
And then for subsequent, you know, occurrences of that storage, we make sure they get all memorized into the same version of the storage.
So okay.
So that's basically in a nutshell how serialization used to work.
in the old days.
And so what happened? So what happened is that we were building so serialization in the old days was just targeted at eager mode.
Right? The only thing people were really serializing were tensors and maybe modules.
Right? Modules with parameters because, you know, Those were just Python objects, but they also had tenses on them.
We we actually discourage people from pickling modules directly, but people do it anyway.
What you're supposed to do is you're supposed get the state, predict for the module and serialize that.
That's because, you know, serializing arbitrary Python objects is kind of error prone.
But anyway, that was what people were normally using serialized in for.
So in comes torch grip.
So what does torch grip need? So torch grip is a bunch of things, but one of the things it is is it's a distribution mechanism for arbitrary byte rich programs, namely torch grip programs that are understood by the torch grip compiler.
And this is important because if you want to sort of ship your models to production, it's important to have a self contained file format that contains all the information you need to run the model.
So the Python code and as well as the tensors.
And so people were, like, looking and they were, like, okay, we need some serialization format.
Fort George script.
And oh, you know, there's this interesting thing which is that Pytorch is using pickle.
But actually, pickle is kind of a bad idea to actually serialize tensors because tensors are really big and you actually wanna, like, if you've got your data living on disc, like, you know, a bunch of parameters.
You wanna just m map them into memory.
You don't wanna actually have to parse them into memory, which is what, you know, traditional PyTorch serialization used to do.
Okay.
So what did they decide? Well, they decided that one, they wanted to use standard file formats.
So we really didn't wanna be in the business of making up a new file format.
because then you don't have any tools that can work with the file format.
And two was, you know, we kinda wanted, you know, our code to, like, be in the Python style.
Right? Like, you know, there's all this existing infrastructure for pickling and unpickling Python objects And, you know, if we define a totally different serialization format rather than pickling, well, we'd have to redo all of that, and then we'd have to keep these in sync indefinitely.
So what did they do? So they did two things.
So one is that they decided we were gonna use zip files for our serialization format.
Don't laugh, zip files are actually really cool.
It's a really well designed file format.
And one of the reasons it's really well designed is, well, you don't actually have to compress things in a zip file.
So you have an uncompressed zip file.
What it turns out is that zip files have a bunch of really good properties.
One is that you don't actually have to read through the entire zip file to figure out where things are.
There's a manifest at the beginning that lets you efficiently index to any particular location.
So if you got a bunch of big tensors, you don't have to scan to all of them to actually find out where your tensor you're interested in is.
Another really good property of the zip file format is that it, you know, is the tenses are laid out exactly as is in memory, so you can easily m map them into memory if you wanna load them in your package process.
And finally, like, everyone knows zip files.
Right? Like, zip files are, you know, the Darling compression format.
in Windows and, like, you know, there are tons and tons of full tools that can work with zip files efficiently.
So if you have a serialized, you know, thing that Pytorch gave you from torch dot save in a recent enough version of Pytorch, you can just unzip it literally.
like unzip it, use it, like rename its file extension to zip, and then like double click it.
And it'll give you all of the internal bits.
The second choice they made was they were gonna keep using pickle.
So what does that mean? Well, remember, I said pickle is a very simple serialization format.
you know, like, most of the complexity involved with it is because, like, you can call arbitrary code to actually reconstitute these objects.
that are safe.
But other than that, you know, you're just saving these two poles of various other things that in themselves, you know, might be two poles of other things.
So what did we do? We just implemented a pickler and unpickler from c plus plus.
So inside jit slash serialization.
There is a pickler and an unpickler, and it is feature for feature compatible with our Python implementation.
and it understands the pickle format.
And this implementation is secure because unlike stock Python Pickler, which, you know, will just attempt to unpickle anything that you throw it, our pickler only supports the limited set of, you know, types.
And all those types don't actually do remote codecs.
execution.
So you know you're safe there.
So, hey, so then that's basically where we are today.
So when you search save and touch load without using the use on you know, use the non zip file format, which which does get called occasionally.
For example, if we serialize a storage just by itself directly using pickle.
We don't use the zip file format.
Just just a fun fact.
But if you are using torch dot save in torch dot load, we give you this zip file.
This zip file contains a data pickle that represents, you know, metadata about the tensor in question, and then it contains, you know, a bunch of files representing the actual data in the tensor.
And this works pretty well.
It's a little slower than the old school pickler, but not that much slower.
And People have been pretty happy about this new serialization format.
Okay.
So that's been a whirlwind tour of serialization in PyTorch.
starting from our humble beginnings as a python pickle extension.
And then two are not so humble endings of also a pickle extension, but, you know, also with the zip file around it.
So I hope this explains a little bit about why our serialization code is kind of complicated and also why whenever you wanna make change to the serialization format.
It's really complicated to do so because of BC and FC and also because you have to edit Python and c plus plus.
But hopefully, if that's something you ever actually need to do, you'll know where to look to figure it out.
That's all I have to say for today.
Talk to you next time.
.
