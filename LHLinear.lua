local LHLinear, parent = torch.class('nn.LHLinear', 'nn.Linear')

function LHLinear:__init(inputSize, outputSize)
   parent.__init(self, inputSize, outputSize)

   self.gradInput = {}
   self.gradInput.grad = torch.Tensor()
   self.gradInput.hessian = torch.Tensor()
end

function LHLinear:updateGradInput(input, gradOutput)
   local gi = self.gradInput.grad
   local hi = self.gradInput.hessian
   local go = gradOutput.grad
   local ho = gradOutput.hessian

   if input:dim() == 1 then
      local inputSize = input:size(1)
      gi:resize(inputSize)
      hi:resize(inputSize, inputSize)

      gi:mv(self.weight:t(), go)
      hi:mm(self.weight:t(), torch.mm(ho, self.weight))

      return self.gradInput
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local inputSize = input:size(2)
      gi:resize(nframe, inputSize)
      hi:resize(nframe, inputSize, inputSize)

      gi:mm(go, self.weight)
      for i=1,nframe do
         hi[i]:mm(self.weight:t(), torch.mm(ho[i], self.weight))
      end

      return self.gradInput
   else
      error('vector or matrix input expected')
   end
end

function LHLinear:accGradParameters(input, gradOutput, scale)
   local scale = scale or 1
   local go = gradOutput.grad
   local ho = gradOutput.hessian

   if input:dim() == 1 then
      -- print(self.weight:size(), torch.max(torch.abs(self.weight)), torch.min(torch.abs(self.bias)))
      local hessianDiag = torch.abs(torch.diag(ho)):add(1)
      local goAdjusted = go:clone():cdiv(hessianDiag)
      -- print(torch.max(tmp), torch.min(tmp))
      self.gradWeight:addr(scale, goAdjusted, input)
      self.gradBias:add(scale, goAdjusted)
   elseif input:dim() == 2 then
      print(self.weight:size(), torch.max(torch.abs(self.weight)), torch.min(torch.abs(self.bias)))
      local nframe = input:size(1)
      local hessianDiag = torch.Tensor():resizeAs(go)
      for i=1,nframe do
         hessianDiag[i] = torch.abs(torch.diag(ho[i])):add(1)
         --print('hessian', hessianDiag:size(), torch.max(hessianDiag[i]), torch.min(hessianDiag[i]))
      end
      local goAdjusted = go:clone():cdiv(hessianDiag)
      self.gradWeight:addmm(scale, goAdjusted:t(), input)
      self.gradBias:addmv(scale, goAdjusted:t(), torch.Tensor(nframe):fill(1))
   else
      error('vector or matrix input expected')
   end
end