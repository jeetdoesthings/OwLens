import SwiftUI

/// Launch & setup splash screen — pure monochromatic, Geist typography.
struct SplashView: View {
    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 6) {
                Text("OwLens")
                    .font(.geist(.bold, size: 44))
                    .foregroundColor(.white)
                    .tracking(2.5)

                Text("RAW LOG CINEMA CAMERA")
                    .font(.geistMono(.regular, size: 10))
                    .foregroundColor(OwLensTheme.textMuted)
                    .tracking(3)
            }

            VStack {
                Spacer()
                HStack(spacing: 6) {
                    Image("github_logo")
                        .resizable()
                        .renderingMode(.template)
                        .aspectRatio(contentMode: .fit)
                        .frame(width: 14, height: 14)
                        .foregroundColor(OwLensTheme.textMuted)
                    Text("jeetdoesthings")
                        .font(.geist(.regular, size: 13))
                        .foregroundColor(OwLensTheme.textMuted)
                        .tracking(0.5)
                }
                .padding(.bottom, 36)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.black)
    }
}

#Preview {
    SplashView()
}
