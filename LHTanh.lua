local LHTanh, parent = torch.class('nn.LHTanh', 'nn.Tanh')

function LHTanh:__init()
   parent.__init(self)

   self.gradInput = {}
   self.gradInput.grad = torch.Tensor()
   self.gradInput.hessian = torch.Tensor()
end

function LHTanh:updateGradInput(input, gradOutput)
   local gi = self.gradInput.grad
   local hi = self.gradInput.hessian
   local go = gradOutput.grad
   local ho = gradOutput.hessian

   if input:dim() == 1 then
      local inputSize = input:size(1)
      gi:resize(inputSize)
      hi:resize(inputSize, inputSize)

      -- This assumes that forward(input) has been called before
      dy = self.output:clone():apply(function(y) return 1-y*y end)
      ddy = self.output:clone():apply(function(y) return -2*y*(1-y*y) end)
      gi:cmul(dy, go)
      hi:mm(torch.diag(dy), torch.mm(ho, torch.diag(dy)))
      local hiDiag = torch.Tensor(hi:storage(), 1, inputSize, inputSize+1)
      hiDiag:addcmul(ddy, go)

      return self.gradInput
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local inputSize = input:size(2)
      gi:resize(nframe, inputSize)
      hi:resize(nframe, inputSize, inputSize)

      dy = self.output:clone():apply(function(y) return 1-y*y end)
      ddy = self.output:clone():apply(function(y) return -2*y*(1-y*y) end)
      gi:cmul(dy, go)
      for i=1,nframe do
         local m = torch.mm(torch.diag(dy[i]), torch.mm(ho[i], torch.diag(dy[i])))
         local mDiag = torch.Tensor(m:storage(), 1, inputSize, inputSize+1)
         mDiag:addcmul(ddy[i], go[i])
         hi[i]:copy(m)
      end

      return self.gradInput
   else
      error('vector or matrix input expected')
   end
end