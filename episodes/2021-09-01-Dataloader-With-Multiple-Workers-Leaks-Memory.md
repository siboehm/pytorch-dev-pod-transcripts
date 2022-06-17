---
layout: post
title: "Dataloader With Multiple Workers Leaks Memory"
date: 2021-09-01
---
The original audio files are available at https://pytorch-dev-podcast.simplecast.com/

# Dataloader With Multiple Workers Leaks Memory

Hello, everyone, and welcome to the Pytorch dev podcast.
Today, I want to talk about a famous bug in Pytorch bug one three two four six.
AKA data loader leaks memory when workers is greater than zero.
This is my apology for not actually knowing how to do a data loader episode because, you know, the subject of data loaders is deep and vast.
and I should probably do a interview with Mittelay Fedunion, our main developer working on data learning right now.
So instead, I'm going to just talk about this particular issue, which is of interest to anyone who's ever, you know, trained any models in Pytorch.
And talk about all the things you need to understand to know exactly what is going on with the issue, why the issue happens, and why the various fixes for it works.
So at the end of this podcast, hopefully, you'll know about all of these things.
So to start, I should explain what exactly this bug looks like from the perspective of the user.
So imagine you're trying to train some model in Pytorch There's a bunch of things that are important to training a model, but in particular, we want to look at how exactly you're getting data into the model in the first place.
Right? Like your data is gonna be some dataset of images or audio files or whatever depending on your domain.
you can somehow load it up into memory and then actually feed it into your model to do the training in question.
And so that process of loading the data is done by the aptly named data loader, which is responsible for, you know, getting this data from wherever it is, doing some preprocessing on it, and then formatting it into Pyturgical Tensors so that we can actually use it for actual, you know, processing.
So the bug looks like this.
So data loader has this feature called num workers which lets you parallelize the data loading process over multiple processes.
This is pretty handy because sometimes you are a CPU bound on the, you know, number of pre processing steps you can do.
And so faring them out to a bunch of, you know, separate processes can help make sure that your actual model stays full of data because maybe your GPUs are actually running way faster than the process of loading your data.
That this is a very easy to accidentally end up in.
And so, like, paralyzing the data loading process can help in this situation.
So what you do is you've got your data loader and you say, okay, I want the number of workers to be, you know, eight or eighteen or however many you think you want your powers and to be.
You start running your model.
It starts training.
Everything looks okay.
You know, it's using a lot of memory, but it's within the bounds of your CPU system.
and, you know, you start doing iterations one after another.
And at some point, you, out of memory, And so you run it again and you look at the memory usage and you notice the memory usage is slowly going up as you are running your training process.
And you're like, there must be some sort of memory leak in the data loader.
And so the issues, original name, was on data loader makes memory when workers is greater than zero.
You'll also notice that if you don't set the number of workers to, you know, something big then the the leak quote unquote doesn't actually happen.
So that's what the bug looks like.
Now, to explain where this bug comes from because actually, in fact, it's not a technically a byte rich problem It's a problem with c Python and it's actually a very difficult problem to resolve at the c Python level.
We have to talk about a lot of concepts.
So one is I need to explain, you know, what exactly is going on with data loading and multiple workers and why we want to do it and how this is set up.
Two, we need to talk a little bit about how process creation works on Linux, what fork is, what copy and right page memory is.
And finally, we have to talk a little bit about the c Python runtime, namely what is the reference counting, and what is it all about.
Eagle eyed listeners, forgive my mixed metaphors here.
May you notice that in fact, we have talked about many of these things in previous versions of the podcast.
I wanna just talk about them over again today because it's important to understanding what is going on with this so called memory leak in data loader.
Okay.
So let's first talk about data set and data loader.
So as I mentioned, data loading is very important for deep learning, training, And sometimes it's hard to make sure that you, you know, have enough data to actually keep your model busy on it.
And so that's why people often wanna do parallelization.
Now, how exactly does parallelization work in Pyturbia's existing data loader design? And this is important because the way we set things up here contributes to the likelihood you'll run into this problem.
So the first thing to remember is that data loader was originally designed to be something that just works in a single process.
So people just, you know, look at it and try to make something that, you know, would be reasonably idiomatic and makes sense if you wanted to load things from a single process.
So the way things tend to work in the dataset is, well, you've got some dataset.
So you need to run some constructor for it, which, you know, initializes some stuff about the dataset.
more on this later.
And then depending on whether or not you're doing one of these interval style or map style data sets, there's some way of actually fetching data when you want to get it in the dataset in question.
So a very common separation, say if you're doing training on an image model, is in the constructor for the dataset, you load up a list of file names, say, for all of the images that, you know, might be in your dataset.
And, you know, that helpful because it can tell you, you know, how long the dataset is and, you know, what are all the possible, like, indexes that you can sample in this situation.
And then when you actually index into the dataset to get something, that's when we actually load data from the image in question.
So so what does this look like? Right? So you, like, have your constructor.
You say, okay.
We'll load up all the file names, store it as a method on the stored as a property on the object, and then inside the iterator for the object, read out that property, do some stuff with it, you know, read out the actual image, give it to the user.
So this is like the obvious way you would go about writing a dataset in a single process case.
And one of the things that Dale already wanted to do was we wanted to, like, keep this same code working, but just on multiple workers.
So how exactly do we do that? Right? Because like we're accessing this data that was constructed in the dataset and, you know, like, what's going on with all the workers in question? Intuitively, what's going on is we actually are able to access these properties on the dataset from each of the workers in question, even though we only allocated them once in the parent process.
So how exactly does that work? To answer that question, we need to know a little bit about how multi processing works, and in particular, how multi processing with fork works.
So Fork is a core primitive in the Munich style operating systems.
And what it does is it takes some process and it makes a copy of it.
Literally a copy.
So that's why we call it a fork because, you know, previously there was one process.
Now there there are two processes and they are exactly the same.
Well, except for the fact that, you know, when you do the four ciscoll, one process gets zero, the other process gets one.
That's how you tell if you're the parent or the child.
Now, this might sound kind of crazy pants.
Right? Like, why would you go through all the trouble of, you know, copying all of the memory from the first process into the next process.
Like, what's up with that? Well, it's kinda useful.
Right? Because maybe there's a bunch of memory that you set up beforehand, and then the code after the fork wants to make use of it.
And so while you need it in the parent process and you need it in child process, and in fact, forking is very cheap in operating systems like Linux, because of an optimization called copy on write.
So remember when I talked about shared memory in PyTorch and I said, hey, you know, normally, each process has its own memory, but in some circumstances, you can share memory between processes, and that's how shared memory CPU tenses work, and that's also how shared libraries in your operating system works well.
Like a single library is loaded up once into physical memory, but then mapped into multiple processes via virtual memory mapping on your operating system.
Well, the same applies when you do a fork So when you do a fork, we don't actually go ahead and eagerly clone all of physical pages.
We just make a copy that refers to the same physical page.
Now, of course, each of the processes that the child and the parent could go ahead and start mutating these pages and the the, like, sort of semantic meaning of a fork is you actually did get a copy.
So if we don't actually make a copy, when someone writes to it, we have to then actually materialize the copy and that's why it's called copy and write.
It's free as long as you only read it and if you start writing into it, well now we're gonna start doing copies on these pages.
So going back to the data loader, well, you know, so what's happening when we have multiple workers is we just fork the Python process.
Every process still gets access to all the stuff that you initialized in the constructor for the dataset.
And as long as you don't write to it, which, you know, like, intuitively, you're not doing any writing to the, you know, like, list of file names, but you're just reading from it.
then, you know, you should be able to share this memory without actually having any problems.
Right? Right? Well, there's one last piece of the puzzle and that's Python reference counting.
In Python, things that look like read only operations like, oh, give me, you know, the field of this object and assign it to a variable, these so called read only objects operations actually do rights under the hood to the memory in question.
And what are these rights for? Therefore, reference counting.
Reference counting is a way of ensuring that we know how many outstanding references there are to any given piece of data so that when the rev count goes to zero, we know we can deallocate it.
What this means is that if you, you know, read some field out of an object and assign it to a new variable, that didn't exist before, we're obligated to increase the reference count of the object in question.
And that's a memory write.
So hopefully, you can see where this is going.
So putting all the pieces together.
So why when we, you know, run the data loader initially, there's not very much memory used.
even though we've spawned off all these workers.
Well, that's because of the fork copy on right optimization, which says is that, hey, when you immediately fork the process we don't need to use that much memory because we can just, you know, share the pages between the processes.
Of course, if we start writing to those pages and that's what happens when Python reference counting comes into play, then you will start actually, you know, writing to the pages and forcing them to be materialized.
And so As you go through your dataset, as you process more and more items, you will start touching more and more reference counts, causing more and more pages to get copied to your child processes.
until in the worst case scenario, every child processes using as much memory as the parent process.
And sure that's not a big deal if your parent process was only using, you know, ten megabytes of memory.
but it is a pretty big deal if your parent process was using four gigabytes of memory and, you know, four times ten worker processes.
That's forty gigabytes.
You're probably out of memory at that point.
So what can you do about this situation? And actually, we can just examine the various, you know, things that led to this problem and each of them sort of suggest a way to solve this issue.
So you might say, hey, the problem is that we're doing this Python reference counting.
And, you know, like, if we had some way of sharing data between processes, without requiring you to increase the reference count when you access them, that would prevent us from paging this copy on right memory into, you know, copies in the child processes and save us from a lot of memory usage.
Well, that's not so easy to do with pure Python objects but it's easy enough to do with other types of objects, like non pie arrays and pie arrow arrays.
These are objects.
They are reference counted per se, but the data in question, each individual integer that's stored in a Numpy array or, you know, as people were doing in, you know, workarounds for this issue, storing strings in Numpy array those things inside of the array themselves are not reference counted.
So as long as you don't actually like take out a new reference to your Numpy array, then you can just, you know, index out a subset of the Numpy array and that will actually just you know, be an operation you can do without incurring any reference count bump.
Of course, even if you actually cause a reference count bump on the numb payer rate, you might still get lucky if, say, the actual data for the number array was allocated out of line, and so you'd, you know, like they lived on different pages.
You only cause one page to come in but not the rest of your data.
Although, I wouldn't count on that.
Just make sure you don't increase the reference counts.
on the shared data you're accessing.
There's a bunch of other things you can do.
Right? Like, you can use c types to allocate raw data.
You can also use any other library that, you know, basically wraps around a raw c representation of the data in question that doesn't involve real Python objects.
Another conceptual fix to this problem is to say, hey, this, you know, accessing of shared memory is kind of, you know, bogus.
Right? Like, The first rule of designing distributed systems is shared memory is bad.
Right? You want explicit queues, you wanna be explicitly about saying what communication you do between processes.
It's a lot easier to debug.
It's a lot more scalable.
It, you know, prevents problems like this.
And so that's what the sort of data loader rewrite that Vitalifa Yunnan and Eugeo Gwan have been working on and specifically the data pipe's concept is that instead of having this monolithic dataset object that, like, does everything that you wanna do, we'll have a bunch of composable data pipes which you can hook up with queues and that do various sets of processing.
The most important thing is it's functional and so you don't actually have any shared state.
Right? Like, when I wanna feed something from one data pipe to another, I have to do it via an explicit queue, and that would prevent this problem as well.
Now, there's one more way of solving this problem, which isn't even mentioned on the issue in question, but which I discovered recently, thanks to some of my colleagues at Facebook.
So another way you could solve this issue is you could literally go into see Python and say, hey, these objects, I just don't want you to increase the reference count anymore.
Right? I wanna somehow make these objects immortal And so, you know, in c Python, if I, you know, access an immortal object, I'm just gonna skip the reference count entirely.
If if you can somehow do that, right, then you could actually use honest goodness normal Python objects in the good old fashioned data loader API.
And that's you know, kind of attractive because it is kind of a pain to go and pack all your strings in an umpire raise.
Well, it turns out There is a fork of the c Python interpreter called cinder developed by folks at Facebook.
I can talk about this because cinder is actually open source.
You can go download it and try it out.
And Tinder implements an API for immortalizing the entirety of your Python heap.
So the way it works is, at some point in time, you can say, hey, I think everything on this heap is going to be live for the rest of eternity.
And Cinder will go ahead and, you know, mark all those objects as a mortal and now you'll no longer do reference counts on them, which means that if you then fork and have workers access that memory, they can access it willy nilly without worrying about reference counts.
So there you have it.
One of the most famous, quote unquote, memory leaks in data loader.
It's probably affected everyone who's done any non trivial processing.
with day loaders in Pytorch.
I'm not going to say that you know Pytorch exactly is blameless here.
Although this is technically a c Python and fork and interaction problem, we probably could have done a better job designing the core abstraction in PyTorch to make it harder to actually accidentally run into this case.
But it's a pretty interesting problem.
One that, you know, is likely to show up if you do any other sort of multi processing.
I hope this was an interesting podcast and gave you a little bit of insight about some of the complexities and interesting internal workings of working with data loaders in PyTorch.
That's everything I wanted to say for today.
Talk to you next time.
.
