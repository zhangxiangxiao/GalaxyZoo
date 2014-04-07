--[[ Tester for Galaxy Zoo
By Xiang Zhang @ New York University
--]]

require("sys")

local Test = torch.class("Test")

-- Initialization of the testing script
-- data: Testing dataset
-- model: Testing model
-- loss: Loss used for testing
-- config: (optional) the configuration table
function Test:__init(data,model,loss,config)
      local config = config or {}

   -- Store the objects
   self.data = data
   self.model = model
   self.loss = loss

   -- Move the type
   self.loss:type(model:type())

   -- Create time table
   self.time = {}

   -- Store configurations
   self.normalize = config.normalize
end

-- Execute testing for a batch step
function Test:run(logfunc)
   -- Initializing the errors and losses
   self.e = 0
   self.l = 0
   self.n = 0
   if self.confusion then self.confusion:zero() end
   
   -- Start the loop
   self.clock = sys.clock()
   for batch,labels,n in self.data:iterator() do
      self.batch = self.batch or batch:type(self.model:type())
      self.labels = self.labels or labels:type(self.model:type())
      self.batch:copy(batch)
      self.labels:copy(labels)
      -- Normalize
      if self.normalize then
	 for i = 1,self.batch:size(1) do
	    self.batch[i]:add(-self.batch[i]:mean()):div(self.batch[i]:std()+1e-8)
	 end
      end
      -- Record time
      if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
      self.time.data = sys.clock() - self.clock
      
      self.clock = sys.clock()
      -- Forward propagation
      self.output = self.model:forward(self.batch)
      self.objective = self.loss:forward(self.output,self.labels)
      if type(self.objective) ~= "number" then self.objective = self.objective[1] end
      -- Record time
      if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
      self.time.forward = sys.clock() - self.clock
      
      self.clock = sys.clock()
      -- Accumulate the errors and losses
      self.l = self.l*(self.n/(self.n+n)) + self.objective*(n/(self.n+n))
      self.n = self.n + n
      -- Record time
      if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
      self.time.accumulate = sys.clock() - self.clock
      
      -- Call the log function
      if logfunc then logfunc(self) end
      
      self.clock = sys.clock()
   end
end
