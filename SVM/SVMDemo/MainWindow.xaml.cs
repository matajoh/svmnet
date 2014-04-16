using SVM;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace SVMDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private const int CLUSTER_SIZE = 10;
        private const int CLUSTER_STDDEV = 8;
        private const int SCALE = 2;

        private Thread _classifyThread;

        private List<DataPoint> _data = new List<DataPoint>();
        private Color[] _colors;
        private Random _rand = new Random();

        public MainWindow()
        {
            InitializeComponent();

            _colors = new Color[classCB.Items.Count];
            Color lighten = Color.FromRgb(120, 120, 120);
            for (int i = 0; i < _colors.Length; i++)
                _colors[i] = ((classCB.Items[i] as ComboBoxItem).Foreground as SolidColorBrush).Color + lighten;
        }

        void classify(object args)
        {
            Problem train = new Problem
            {
                X = _data.Select(o => new Node[] { new Node(1, o.Position.X), new Node(2, o.Position.Y) }).ToArray(),
                Y = _data.Select(o => o.Label).ToArray(),
                Count = _data.Count,
                MaxIndex = 2
            };
            Parameter param = args as Parameter;

            RangeTransform transform = RangeTransform.Compute(train);
            train = transform.Scale(train);

            Model model = Training.Train(train, param);

            int width = (int)plot.ActualWidth;
            int height = (int)plot.ActualHeight;
            byte[] pixels = new byte[width * height * 3];          

            int cWidth = (width >> SCALE) + 1;
            int cHeight = (height >> SCALE) + 1;
            int[,] labels = new int[cHeight, cWidth];
            for(int r=0, i=0; r<cHeight; r++)
                for (int c = 0; c < cWidth; c++, i++)
                {
                    int rr = r << SCALE;
                    int cc = c << SCALE;
                    Node[] datum = new Node[] { new Node(1, cc), new Node(2, rr) };
                    datum = transform.Transform(datum);
                    labels[r, c] = (int)model.Predict(datum);
                    classifyPB.Dispatcher.Invoke(() => classifyPB.Value = (i * 100) / (cHeight * cWidth));
                }
            
            PixelFormat format = PixelFormats.Rgb24;
            for (int i = 0, r = 0; r < height; r++)
            {
                for (int c = 0; c < width; c++)
                {
                    int label = labels[r >> SCALE, c >> SCALE];
                    Color color = _colors[label];
                    pixels[i++] = color.R;
                    pixels[i++] = color.G;
                    pixels[i++] = color.B;                    
                }
            }

            plot.Dispatcher.Invoke(() =>
            {
                ImageBrush brush = new ImageBrush(BitmapSource.Create(width, height, 96, 96, format, null, pixels, width * 3));
                brush.Stretch = Stretch.None;
                brush.AlignmentX = 0;
                brush.AlignmentY = 0;
                plot.Background = brush;
            });

            classifyPB.Dispatcher.Invoke(() => classifyPB.Value = 0);
        }

        private void classCB_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            classCB.Foreground = (classCB.SelectedItem as ComboBoxItem).Foreground;
        }

        private void plot_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            addPoints(e.GetPosition(plot), classCB.SelectedIndex);
        }

        private void classifyB_Click(object sender, RoutedEventArgs e)
        {
            if (_data.Count == 0)
                return;

            Parameter param = new Parameter();
            param.Gamma = .5;
            param.SvmType = (SvmType)svmTypeCB.SelectedIndex;
            param.KernelType = (KernelType)kernelTypeCB.SelectedIndex;

            _classifyThread = new Thread(new ParameterizedThreadStart(classify));
            _classifyThread.Start(param);
        }

        private void plot_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            int label = (classCB.SelectedIndex + 3) % classCB.Items.Count;
            addPoints(e.GetPosition(plot), label);
        }

        private void addDataPoint(Point position, int label)
        {
            DataPoint datum = new DataPoint { Position = position, Label = label };
            _data.Add(datum);

            Ellipse ellipse = new Ellipse();
            ellipse.Fill = (classCB.Items[label] as ComboBoxItem).Foreground;
            ellipse.Width = 8;
            ellipse.Height = 8;
            ellipse.Stroke = new SolidColorBrush(Colors.Black);
            ellipse.StrokeThickness = 1;
            Canvas.SetLeft(ellipse, datum.Position.X - 4);
            Canvas.SetTop(ellipse, datum.Position.Y - 4);
            plot.Children.Add(ellipse);
        }

        private void addPoints(Point center, int label)
        {
            addDataPoint(center, label);
            if (placementCB.SelectedIndex == 1)
            {
                for (int i = 0; i < CLUSTER_SIZE; i++)
                {
                    Point sample;

                    do
                    {
                        sample = samplePoint(center, CLUSTER_STDDEV);
                    }
                    while (sample.X < 0 || sample.X >= plot.ActualWidth || sample.Y < 0 || sample.Y >= plot.ActualHeight);

                    addDataPoint(sample, label);
                }
            }
        }

        private void clearB_Click(object sender, RoutedEventArgs e)
        {
            _data.Clear();
            plot.Background = new SolidColorBrush(Colors.White);
            plot.Children.Clear();
        }

        private Point samplePoint(Point center, double standardDeviation)
        {
            double theta = 2 * Math.PI * _rand.NextDouble();
            double rho = Math.Sqrt(-2 * Math.Log(1 - _rand.NextDouble()));
            double scale = standardDeviation * rho;
            return new Point(center.X + scale * Math.Cos(theta), center.Y + scale * Math.Sin(theta));
        }
    }
}
