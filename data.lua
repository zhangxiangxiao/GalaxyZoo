--[[
Data reader for Galaxy Zoo
By Xiang Zhang @ New York University
--]]

require("image")

-- Create class
local Data = torch.class("Data")

-- Constants
Data.height = 256
Data.width = 256
Data.nclasses = 37

-- Initializer
-- config: configuration table
--    .file: the data index file
--    .root: (optional) overriding data root directory
--    .crop: (optional) a table with .width and .height indicating cropping size (middle crop used)
--    .batch_size: (optional) the size of the minibatch. Default 128.
function Data:__init(config)
   -- Reading index file
   self.file = config.file
   self.data = torch.load(config.file)

   -- Overriding optional roots
   self.data.root = config.root or self.data.root

   -- Store default options
   self.crop = config.crop
   self.scale = config.scale
   self.rotate = config.rotate
   self.parallel = config.parallel
   self.batch_size = config.batch_size or 128

   -- Store for parallel
   self.config = config
end

-- Return one raw image data and format it
-- file: the file name string
-- cropped: (optional) the cropped data
function Data:getImage(ind,index,cropped)
   -- Load the image
   local im
   if ind.data then
      im = ind.data:select(1,index):double():div(255)
   else
      im = image.load(paths.concat(ind.root,ind.files[i]))
   end

   -- Check channels
   if im:dim() == 2 then
      new_im = torch.Tensor(3,im:size(1),im:size(2))
      for c = 1,3 do
	 new_im:select(1,c):copy(im)
      end
      im = new_im
   elseif im:size(1) == 1 then
      new_im = torch.Tensor(3,im:size(2),im:size(3))
      for c = 1,3 do
	 new_im:select(1,c):copy(im:select(1,1))
      end
      im = new_im
   end
   if im:dim() ~= 3 or im:size(1) ~= 3 then
      error("Image channels is not 3")
   end

   -- Do rotation
   if self.rotate then
      local radius = self.rotate*math.random()
      im = image.rotate(im,radius)
   end

   -- Do scale
   if self.scale then
      local scale = (self.scale-1/self.scale)*math.random() + 1/self.scale
      im = image.scale(im, math.floor(scale*im:size(3)), math.floor(scale*im:size(2)))
   end

   -- Do cropping
   local cropped = cropped
   if self.crop and (self.crop.height ~= im:size(2) or self.crop.width ~= im:size(3)) then
      cropped = cropped or torch.Tensor(3,self.crop.height,self.crop.width)
      if self.crop.method == "center" or self.crop.method == nil then
	 local startx = math.floor((im:size(3) - self.crop.width)/2) + 1
	 local starty = math.floor((im:size(2) - self.crop.height)/2) + 1
	 local endx = startx + self.crop.width - 1
	 local endy = starty + self.crop.height - 1
	 cropped:copy(im[{{},{starty,endy},{startx,endx}}])
      elseif self.crop.method == "random" then
	 local startx = torch.random(im:size(3) - self.crop.width + 1)
	 local starty = torch.random(im:size(2) - self.crop.height + 1)
	 local endx = startx + self.crop.width - 1
	 local endy = starty + self.crop.height - 1
	 cropped:copy(im[{{},{starty,endy},{startx,endx}}])
      else
	 error("Unreconized cropping method")
      end
   else
      if cropped == nil then
	 cropped = im
      else
	 cropped:copy(im)
      end
   end

   -- Return the data
   return cropped
end

-- Return random batch set of images
-- inputs: (optional) inputs object. can be nil.
-- labels: (optional) labels object. can be nil.
-- ind: (optional) index object of the set
function Data:getBatch(inputs,labels,ind)
   local ind = ind or self.data
   local inputs = inputs
   if inputs == nil and self.crop then
      inputs = torch.Tensor(self.batch_size,3,self.crop.height,self.crop.width)
   elseif inputs == nil then
      inputs = torch.Tensor(self.batch_size,3,Data.height,Data.width)
   end
   local labels = labels or torch.Tensor(inputs:size(1),Data.nclasses)

   -- Get a random set of images
   for i = 1,inputs:size(1) do
      local index = torch.random(#ind.files)
      self:getImage(ind,index,inputs:select(1,i))
      if ind.labels then labels[i]:copy(ind.labels[index]) end
   end

   -- Return the set
   return inputs, labels
end

-- Return the iterator
-- static: whether to return static data. Default is true
-- ind: the index file
function Data:iterator(static,ind)
   local i = 1
   local ind = ind or self.data
   local inputs,labels
   local static
   if static == nil then static = true end
   if static and self.crop then
      -- Create data
      inputs = torch.Tensor(self.batch_size,3,self.crop.height,self.crop.width)
      labels = torch.Tensor(inputs:size(1),Data.nclasses)
   elseif static then
      inputs = torch.Tensor(self.batch_size,3,Data.height, Data.width)
      labels = torch.Tensor(inputs:size(1),Data.nclasses)
   end

   return function()
      -- Check for end of iteration
      if ind.files[i] == nil then return end
      
      -- Create data if not exist
      local inputs = inputs
      if inputs == nil then
	 if self.crop then
	    inputs = torch.Tensor(self.batch_size,3,self.crop.height,self.crop.width)
	 else
	    inputs = torch.Tensor(self.batch_size,3,Data.height,Data.width)
	 end
      end
      local labels = labels or torch.Tensor(input:size(1), Data.nclasses)
      
      -- Get next set of images
      local n = 0
      for k = 1,inputs:size(1) do
	 if ind.files[i] == nil then
	    break
	 end
	 n = n + 1
	 self:getImage(ind,i,inputs:select(1,k))
	 if ind.labels then labels[k]:copy(ind.labels[i]) end
	 i = i + 1
      end
      
      -- Return the set
      return inputs,labels,n
   end
end
