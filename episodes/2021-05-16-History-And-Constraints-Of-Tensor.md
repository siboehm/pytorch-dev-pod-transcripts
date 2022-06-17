---
layout: post
title: "History And Constraints Of Tensor"
date: 2021-05-16
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# History And Constraints Of Tensor

Hi.
My name is Edward, and welcome to today's edition of the Pyturgical Dev podcast.
Today, I wanna talk about a topic which was requested also multiple times by several people.
namely the history behind tensor, tensor impulse, storage, storage, and basically, like, how is the tensor data structure and Pytorch put together? This is a topic that I have written about in the past.
For example, on my blog, I have a blog post about, you know, basics about Pytuch Internals.
And some of the things that talks about are how tenses are put together.
So, like, there are these things called strides.
You know, we have a concept called storage.
So if you wanna know more about these topics, go check out my blog post, then come up back to this podcast.
Today, I wanna talk a little bit more about some of the historical and design constraints that have led us to where the tensor data structure is today.
So basically, given all these design constraints, if you, you know, spend enough time, hopefully, you would end up in the same situation that Pytorch is.
So I I sort of there are a lot of things in Tensor.
Right? because it's a very traffic data structure a lot of people have added things over the years and sometimes it can be a bit bewildering like why the heck are there like eight bit fields for like various you know, variations of, you know, memory layout on the tensor.
Well, you know, hopefully knowing a little bit about the background and the constraints, will help you understand, oh yeah, I see why that's there.
It might not be ideal, but there is a constraint that causes to get there.
So let's get to it.
So the first and foremost constraint that fed into Pytour's design of Tensor is the fact that Pytour's descends from t h.
I've said this before, I'll say it again.
Remember, Pytorch was originally just python bindings to the pre existing c libraries that shipped with Lewis torch, which in turn came from the torch seven libraries.
And why is this important? Well, we inherited a lot of the basic architecture for tensors from these libraries.
And in particular, the split between tensor and storage is the sort of most prominent thing that, you know, Pytourish carries in his DNA today.
I didn't ever get a chance to talk to any of the original lush or torch seven authors I don't really know why they set things up this way.
But when I sort of like retroactively look at the past and like come up with my own explanations, One thing that I can say is that Pytorch's concept of a storage was very important for, you know, enabling something that's very core to Pytorch's DNA.
namely the ability to alias tenses together and do mutations on them.
This is like very unusual.
The strides especially are very unusual.
Many Many other systems, TensorFlow being one prominent one only support operations on contiguous tensors.
And sort of like what makes Pytorch a little spicy here is that, you know, you can actually, you know, refer to multiple tenses on the same memory possibly with different layouts simply by adjusting the striding.
So it's something that's very like unique to Pytorch and we got that from the libraries that we descended from.
There's other things that we inherited from the THD's as well.
for example, when Tensor was just the c struct and t h, they needed some way to do a reference count.
So they just put the reference count on the Tensor itself.
It turns out that obtrusively rough counting in its way is very convenient, for example, when you're writing bindings.
Because if you have a raw pointer to a object you don't have to, like, do any work with, say, enable shared from this to get out a owning pointer to it.
Right? You can just transmute the owning pointer transmit the raw pointer into an owning pointer and, you know, the owning pointer will just take care of incrementing and decrementing the rough count.
So when, you know, we brought Pytorch into, you know, the c plus plus land and we implemented the classes, we also preserved intrusive rough counting because all of our binding code was way simpler when we had it that way.
Also, we didn't want pointers to tensers to be two words, which is, you know, what shared pointer does in c plus plus.
The second constraint, which is useful to know about on Tensor is the fact that it actually is the result of merging the Cafe two and Pyturg libraries together.
So if you're a regular Pyturg user, you might not, you know, think very much about fei two.
Right? It's this other library that, you know, is graph mode only.
But in fact, the same tensor representation in PyTorch is used verbatim with Cafe two.
There's actually two separate user facing classes.
There is a tensor class that, you know, an AT Tensor class which you use from Pytorch and a cafe two Tensor class that you use from Cafe two.
And they actually have different public APIs for backwards compatibility reasons.
But the both of these are what we call pimple classes, pointer to implementation classes.
So they don't actually you know, represent the data in the object and said they just contain an owning pointer to the tensor in pole object which is the actual object that contains all the data in question.
By the way, why is there this split between tensor and tensor and pull? Well, it's because, you know, we are a Python project.
And a lot of people when writing code involving tensors in c plus plus expect Python style reference semantics to work.
So, like, if I have a tensor y and then I say tensor x equals y, I expect x to, you know, point to the same tensor as y.
I don't expect a copy to happen in this case.
And, you know, in c plus plus value semantics, you know, if you have a value type like Tensor Imple is, you did this copy construction that would actually copy all the metadata in question.
and then it depends on the semantics of the smart pointers inside what the other data does.
So by splitting those into two types and having Tensor be a actual pointer type, like in the same way shared pointer is, you would just write Tensor and you, you know, can assign things around.
and it looks just like how things are in Python.
So, you know, constraint three, I would say, is that, you know, we're our Python project.
So a lot of our c plus plus design comes out of trying a model off of Python.
There's a great essay about this, by the way, which is on the Wiki, basically a manifesto about writing Python in c plus plus.
We yeah.
as time has gone on for efficiency reasons, we have had to walk back some of the things we've done here.
For example, you know, passing around tensors as a pointer type is not so great because they force ref count bumps.
Right? And Python, this is not a big deal because Python has a gill, so the ref counts are non atomic.
automics are kind of expensive.
So, you know, we've actually spent some time in the recent past trying to, you know, remove as many rough cuts as possible.
But generally speaking, if you can write Python code, you can write Pytorch code, and the Tensor Class APIs are designed to make these look as similar as possible.
Okay, point four.
So I'm done with the historical things, but point four is we don't really want there to be virtual calls on Tensor.
And this actually has some pretty major implications.
Now, I should preface this by saying, if you actually go and look at the TensorFlow class, and look at all the methods on it, Actually, a ton of them are virtual.
And there's there's a reason for this.
It's historical reason.
But the reason why we don't wanna virtualize most methods on Tensor Imple is because virtual methods thwart the inliner.
So, you know, most operations on Tensor like, for example, querying the sizes, should compile into a direct, you know, memory access at the field that contains the sizes and questions.
Right? It should be super fast we should be able to get rid of all of the function called Goop.
But, you know, if it's a virtual method while some sub class could have overwritten the behavior in this case, And so we can't in line in this situation.
We have to actually do the v call jump.
And the v call jumps are not that expensive, but you know, we call size everywhere in PyTorch, so it really does add up.
Why is size actually virtual then? Well, you know, this is a sort of like argument between, like, you know, his history and sort of design in the Pyturg Court place.
The history of Pyturg is that size was virtual because when Zach originally wrote the class, it was virtual.
And why was it virtual? Well, it was virtual because we had this variable thing.
seen my previous podcast about the life and death and variable.
We had variable variables of wrapper on a tensor, and they made this very reasonable at the time designed as that they didn't want to duplicate the size information between variable and the tensor that it wrapped.
Because, you know, if you duplicate the information, it can get out of sync.
for example, if you resize the underlying tensor without, you know, telling the variable about it.
So if you don't wanna keep them in sync, you need to change the behavior.
Right? On a tensor, you can just access the field directly.
But on a variable, you have to jump to the base class and then actually query this size there.
So, okay, size is too virtual.
Now, we've gotten rid of variable.
Right? The variable tends to merge.
And so this this constraint no longer applies.
And now we have a design that we can actually just force everyone to, like, accurately record what the size of their sensor is inside the class itself.
But in the meantime, a bunch of people, like, went ahead and overrode size for their own needs.
And so we have to, like, unwind that situation, solve the problem.
most notably XLA, cough, cough.
Okay.
So but, you know, in general, we want methods on Tensor to be virtual.
And what that means is that actually when you look at the Tensor Info class, it basically has all of the fields that you can conceivably want to describe, you know, what a tensor should be.
So for example, we have sizes on tensors.
Yes.
Hypothetically, you know, strange extensions to tensors, like rugged sensors, or NASA tensors, might not have size in the traditional sense.
But, you know, because size is such a, you know, intrinsic operation that we use everywhere in PyTorch, we really do want you to, like, have some, you know, conventional notion of size for anything you model in this way.
And if you can't model in this way, well, maybe, you know, tensor and pull is not for you.
Another consequence of this is exactly those bit feels about memory layout.
Right? Like, we don't want to actually have to compute the memory layout every time.
So, you know, given that we know what the sizes and strides of a tensor are, that actually tells us what the memory layout is.
And so we pre compute a lot of information in these bit fields so that, you know, we can have fast accesses that don't involve doing some compute.
They just, like, check what the bit is.
Okay.
Point five.
Point five is extensibility.
So, you know, tensor is actually, this is the same as the previous point.
Right? Which is that, like, the virtualization constraint is in tension with the extensibility constraint.
Right? By de virtualizing the Tensor Imple class, it's less extensible, but operations on it are faster.
By virtualizing it, you can override more behavior but then the Tensor Imple class is less efficient.
So we kinda need to play this game and so like the the cut we have, right, is that we want basic operations like the basic data model for Tensor to be virtual.
But then anything else on top like especially operators that can all be virtual and in fact it is via the dispatcher.
Okay.
Last constraint.
Size and memory.
I have a really funny story, which is when we were merging the Cafe two and Pytorch libraries.
I added a bunch of fields sort of randomly because, like, I was once again unioning the behavior of Cafe two in Pikeurch.
And then I broke some internal workflows.
And what those internal workflows were doing they were like allocating four million tensors.
And so every word I added to PyTorch actually ballooned their memory usage by several gigabytes.
So that was not very nice.
And it, like, induced us to, like, spend a bunch of time trying to optimize the actual memory size of the tensor embolstrict itself.
because it's it's really overhead.
Right? Like in PyTorch, you really want to just be, you know, storing memory for all of the, you know, actual data that you're doing your deep learning on.
and you don't wanna waste time or space on the metadata in Tensor itself.
And so we've done a bunch of optimizations some very recently, for example, done by Scott Wachok.
For example, we used to store sizes and strides as out of line vectors.
on tensor, that's really inefficient because a standard vector in c plus plus takes three words in the structured salt.
Right? It takes a size, it takes a pointer to the beginning, and it takes the pointer to the end of the reserve data.
So because, you know, vectors can have a size that is less than the actual data that's allocated for it.
So all that needs to be stored and it's not really necessary.
And also, you don't need to serve the sides for both sizes and strides because the dimensionality of a tensor is fixed.
So, you know, we actually pack these fields and we also put the sizes and strides directly in the tensor implant itself assuming that most tensors are five dimensional or smaller.
And that saves us having to do dynamic allocations when we allocate tensors.
Okay.
So that's it for, you know, why Tensor is the way it is.
So the next time you go and look at the Tensor Imple class, hey, think about, you know, well, we wanted this to look like Python.
So that's why there's a pimple method.
We wanted, you know, to support all the stuff we could support from the good old torch days, so that's why there's storage and tensor.
We merged Kafe two and PyTorch together, so that's why there's a bunch of really random features in ten are in full that don't make that much sense.
Well, that's because some of them came from Cafe Doo.
Another example of that is type meta, which, you know, there's two, like, ways of representing d types in plus plus scaler type, which is just an enom and type meta, which is a pointer type that is open and extensible.
And that's because cave two supported registering custom types to tensors like stood strings.
You could have a tensor full of stood strings.
Don't ask me why you'd want it.
Actually, it's pretty useful in some situations.
And then fourth, there's a bunch of, you know, constraints about, like, you know, efficiency.
Right? Like, making sure that our methods can inline.
making sure that the memory size of tensor people isn't too big, but also at the same time supporting extensibility for, you know, all of the weird and wacky other tensor types.
like sparse tensors and nested tensors and, you know, funcatorch tinctures that people want to support.
Okay.
That's everything I wanted to state for today.
Talk to you next time.
.
